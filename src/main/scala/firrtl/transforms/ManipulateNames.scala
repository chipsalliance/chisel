// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms

import firrtl._
import firrtl.analyses.InstanceKeyGraph
import firrtl.Mappers._

import firrtl.annotations.{
  CircuitTarget,
  CompleteTarget,
  InstanceTarget,
  ModuleTarget,
  MultiTargetAnnotation,
  ReferenceTarget,
  Target,
  TargetToken
}
import firrtl.options.Dependency
import firrtl.stage.Forms
import firrtl.stage.TransformManager.TransformDependency

import scala.collection.mutable
import scala.reflect.ClassTag

/** Base trait for annotations that control the behavior of transforms that sub-class ManipulateNames
  * @see [[ManipulateNamesBlocklistAnnotation]]
  * @see [[ManipulateNamesAllowlistAnnotation]]
  * @define noteLocalTargets All targets must be local. Name modification in a non-local target (e.g., a node in a
  * specific instance) makes no structural modification and will be ignored during deduplication. If you want this
  * behavior, use a combination of a sub-class of this annotation and a [[firrtl.transforms.NoDedupAnnotation
  * NoDedupAnnotation]].
  */
sealed trait ManipulateNamesListAnnotation[A <: ManipulateNames[_]] extends MultiTargetAnnotation {

  def transform: Dependency[A]

  /* Throw an exception if targets are non-local */
  targets.flatten.collect {
    case a if !a.isLocal => a
  } match {
    case Nil =>
    case a =>
      val aString = a.map(_.serialize).mkString("\n    - ", "\n    - ", "")
      throw new IllegalArgumentException(s"""'${this.getClass.getName}' given non-local targets: $aString""")
  }

}

/** Annotation to prevent name manipulation of[[firrtl.annotations.Target Target]]s in a transform that subclasses
  * [[ManipulateNames]]. All listed targets will not be modified.
  *
  * @param targets FIRRTL IR targets to exclude from name manipulation
  * @param transform the transform that this should apply to
  * @tparam A a sub-type of [[ManipulateNames]]
  * @throws java.lang.IllegalArgumentException if any non-local targets are given
  * @note $noteLocalTargets
  */
case class ManipulateNamesBlocklistAnnotation[A <: ManipulateNames[_]](
  targets:   Seq[Seq[Target]],
  transform: Dependency[A])
    extends ManipulateNamesListAnnotation[A] {

  override def duplicate(a: Seq[Seq[Target]]) = this.copy(targets = a)

}

/** Annotation to filter name manipulation to only manipulate specific [[firrtl.annotations.Target Target]]s in a
  * transform that subclasses [[ManipulateNames]]. Targets will be renamed if they are not listed in a
  * [[ManipulateNamesBlocklistAnnotation]] and also listed in this annotation.
  *
  * Not providing a [[ManipulateNamesAllowlistAnnotation]] means that all targets in a circuit may be renamed.
  *
  * @param targets FIRRTL IR targets to include in name manipulation
  * @param transform the transform that this should apply to
  * @tparam A a sub-type of [[ManipulateNames]]
  * @throws java.lang.IllegalArgumentException if any non-local targets are given
  * @note $noteLocalTargets
  */
case class ManipulateNamesAllowlistAnnotation[A <: ManipulateNames[_]](
  targets:   Seq[Seq[Target]],
  transform: Dependency[A])
    extends ManipulateNamesListAnnotation[A] {

  override def duplicate(a: Seq[Seq[Target]]) = this.copy(targets = a)

}

/** Records the result of name changes for any targets included in a [[ManipulateNamesAllowlistAnnotation]]
  *
  * If targets are later removed, then a target and old target will be removed from this annotation. If all targets are
  * removed, then this annotation will be deleted.
  *
  * @param targets the new targets
  * @param transform the transform that performed this rename
  * @param oldTargets the old targets
  */
case class ManipulateNamesAllowlistResultAnnotation[A <: ManipulateNames[_]](
  targets:    Seq[Seq[Target]],
  transform:  Dependency[A],
  oldTargets: Seq[Seq[Target]])
    extends MultiTargetAnnotation {

  override def duplicate(a: Seq[Seq[Target]]) = this.copy(targets = a)

  override def update(renames: RenameMap) = {
    val (targetsx, oldTargetsx) = targets.zip(oldTargets).foldLeft((Seq.empty[Seq[Target]], Seq.empty[Seq[Target]])) {
      case ((accT, accO), (t, o)) =>
        t.flatMap(renames(_)) match {
          /* If the target was deleted, delete the old target */
          case tx if tx.isEmpty => (accT, accO)
          case tx               => (Seq(tx) ++ accT, Seq(o) ++ accO)
        }
    }
    targetsx match {
      /* If all targets were deleted, delete the annotation */
      case Nil => Seq.empty
      case _   => Seq(this.copy(targets = targetsx, oldTargets = oldTargetsx))
    }
  }

  /** Return [[firrtl.RenameMap RenameMap]] from old targets to new targets */
  def toRenameMap: RenameMap = {
    val m = oldTargets
      .zip(targets)
      .flatMap {
        case (a, b) => a.map(_ -> b)
      }
      .toMap
      .asInstanceOf[Map[CompleteTarget, Seq[CompleteTarget]]]
    RenameMap.create(m)
  }

}

/** A datastructure used to do single-pass name manipulation
  * @param circuit the [[ir.Circuit]] that will be manipulated
  * @param renames a rename map
  * @param block a function that returns true if a [[firrtl.annotations.Target Target]] should not be renamed
  * @param allow a function that returns true if a [[firrtl.annotations.Target Target]] should be renamed
  */
private class RenameDataStructure(
  circuit:     ir.Circuit,
  val renames: RenameMap,
  val block:   Target => Boolean,
  val allow:   Target => Boolean) {

  /** A mapping of targets to associated namespaces */
  val namespaces: mutable.HashMap[CompleteTarget, Namespace] =
    mutable.HashMap(CircuitTarget(circuit.main) -> Namespace(circuit))

  /** Wraps a HashMap to provide better error messages when accessing a non-existing element */
  class InstanceHashMap {
    type Key = ReferenceTarget
    type Value = Either[ReferenceTarget, InstanceTarget]
    private val m = mutable.HashMap[Key, Value]()
    def apply(key: ReferenceTarget): Value = m.getOrElse(
      key, {
        throw new FirrtlUserException(
          s"""|Reference target '${key.serialize}' did not exist in mapping of reference targets to insts/mems.
              |    This is indicative of a circuit that has not been run through LowerTypes.""".stripMargin
        )
      }
    )
    def update(key: Key, value: Value): Unit = m.update(key, value)
  }

  /** A mapping of a reference to either an instance or a memory (encoded as a [[ReferenceTarget]] */
  val instanceMap: InstanceHashMap = new InstanceHashMap

  /** Return true if a target should be skipped based on allow and block parameters */
  def skip(a: Target): Boolean = block(a) || !allow(a)

}

/** Transform for manipulate all the names in a FIRRTL circuit.
  * @tparam A the type of the child transform
  */
abstract class ManipulateNames[A <: ManipulateNames[_]: ClassTag] extends Transform with DependencyAPIMigration {

  /** A function used to manipulate a name in a FIRRTL circuit */
  def manipulate: (String, Namespace) => Option[String]

  override def prerequisites:          Seq[TransformDependency] = Seq(Dependency(firrtl.passes.LowerTypes))
  override def optionalPrerequisites:  Seq[TransformDependency] = Seq.empty
  override def optionalPrerequisiteOf: Seq[TransformDependency] = Forms.LowEmitters
  override def invalidates(a: Transform) = a match {
    case passes.InferTypes | _: analyses.GetNamespace => true
    case _ => false
  }

  /** Compute a new name for some target and record the rename if the new name differs. If the top module or the circuit
    * is renamed, both will be renamed.
    * @param name the string to rename
    * @param r a data structure containing information necessary for renaming
    * @param target the target associated with the name
    * @return a new name (or the old name if no renaming is necessary)
    */
  private def doRename(name: String, r: RenameDataStructure, target: CompleteTarget): String = {
    /* Compute the new name and, if the name is a new name, a new target. */
    val (namex: String, ax: Option[CompleteTarget]) = target match {
      /* Do not rename if this is designated as a skip */
      case a if r.skip(a) =>
        (name, None)
      /* Circuit renaming */
      case a @ CircuitTarget(b) =>
        manipulate(b, r.namespaces(a)) match {
          case Some(str) => (str, Some(a.copy(circuit = str)))
          case None      => (b, None)
        }
      /* Module renaming for non-top modules */
      case a @ ModuleTarget(_, b) =>
        manipulate(b, r.namespaces(a.circuitTarget)) match {
          case Some(str) => (str, Some(a.copy(module = str)))
          case None      => (b, None)
        }
      /* Instance renaming */
      case a @ InstanceTarget(_, _, Nil, b, c) =>
        manipulate(b, r.namespaces(a.moduleTarget)) match {
          case Some(str) => (str, Some(a.copy(instance = str)))
          case None      => (b, None)
        }
      /* Rename either a module component or a memory */
      case a @ ReferenceTarget(_, _, _, b, Nil) =>
        manipulate(b, r.namespaces(a.moduleTarget)) match {
          case Some(str) => (str, Some(a.copy(ref = str)))
          case None      => (b, None)
        }
      /* Rename an instance port or a memory reader/writer/readwriter */
      case a @ ReferenceTarget(_, _, _, b, (token @ TargetToken.Field(c)) :: Nil) =>
        val ref = r.instanceMap(a.moduleTarget.ref(b)) match {
          case Right(inst) => inst.ofModuleTarget
          case Left(mem)   => mem
        }
        manipulate(c, r.namespaces(ref)) match {
          case Some(str) => (str, Some(a.copy(component = Seq(token.copy(str)))))
          case None      => (c, None)
        }
    }
    /* Record the optional rename. If the circuit was renamed, also rename the top module. If the top module was
     * renamed, also rename the circuit. */
    ax.foreach(axx =>
      target match {
        case c: CircuitTarget =>
          r.renames.rename(target, r.renames(axx))
          r.renames.rename(c.module(c.circuit), CircuitTarget(namex).module(namex))
        /* Note: this code path is not exercised by the implementation of the [[run]] and [[onModule]] methods. Those
         * only use [[doRename]] on the circuit and [[maybeRename]] on the top module.
         */
        case m: ModuleTarget if m.module == m.circuit =>
          r.renames.rename(target, r.renames(axx))
          r.renames.rename(m.circuitTarget, axx.circuitTarget)
        case _ =>
          r.renames.rename(target, r.renames(axx))
      }
    )
    namex
  }

  /** Change a name based on known renames. Do not record any new renames.
    * @param name the string to rename
    * @param r a data structure containing information necessary for renaming
    * @param target the target associated with the name
    * @return a new name (or the old name if no renaming is necessary)
    */
  private def maybeRename(name: String, r: RenameDataStructure, t: CompleteTarget): String =
    r.renames.underlying.get(t) match {
      case Some(ax) if ax.size == 1 =>
        ax match {
          case Seq(foo: CircuitTarget) => foo.name
          case Seq(foo: ModuleTarget) => foo.module
          case Seq(foo: InstanceTarget) => foo.instance
          case Seq(foo: ReferenceTarget) =>
            foo.tokens.last match {
              case TargetToken.Ref(value)   => value
              case TargetToken.Field(value) => value
              case _ =>
                Utils.throwInternalError(
                  s"""|Reference target '${t.serialize}'must end in 'Ref' or 'Field'
                  |    This is indicative of a circuit that has not been run through LowerTypes.""",
                  Some(new MatchError(foo.serialize))
                )
            }
        }
      case s @ Some(ax) =>
        Utils.throwInternalError(
          s"""Found multiple renames '${t}' -> [${ax.map(_.serialize).mkString(",")}]. This should be impossible.""",
          Some(new MatchError(s))
        )
      case None => name
    }

  /** Rename an expression
    *
    * This logic exploits the known structure of the output of [[LowerTypes]] such that the only possible expressions in
    * a module are: (1) references to module components, (2) subfields of references are instance components, and (3)
    * subfields of subfields or references are memory ports.
    */
  private def onExpression(e: ir.Expression, r: RenameDataStructure, t: ModuleTarget): ir.Expression = e match {
    /* A reference to something inside this module */
    case w: WRef => w.copy(name = maybeRename(w.name, r, Target.asTarget(t)(w)))
    /* This is either the subfield of an instance or a subfield of a memory reader/writer/readwriter */
    case w @ WSubField(expr, ref, _, _) =>
      expr match {
        /* This is an instance */
        case we @ WRef(inst, _, _, _) =>
          val tx = Target.asTarget(t)(we)
          val (rTarget: ReferenceTarget, iTarget: InstanceTarget) = r.instanceMap(tx) match {
            case Right(a) => (a.ofModuleTarget.ref(ref), a)
            case a @ Left(ref) =>
              throw new FirrtlUserException(
                s"""|Unexpected '${ref.serialize}' in instanceMap for key '${tx.serialize}' on expression '${w.serialize}'.
                |    This is indicative of a circuit that has not been run through LowerTypes.""",
                new MatchError(a)
              )
          }
          w.copy(we.copy(name = maybeRename(inst, r, iTarget)), name = maybeRename(ref, r, rTarget))
        /* This is a reader/writer/readwriter */
        case ws @ WSubField(expr, port, _, _) =>
          expr match {
            /* This is the memory. */
            case wr @ WRef(mem, _, _, _) =>
              w.copy(
                expr = ws.copy(
                  expr = wr.copy(name = maybeRename(mem, r, t.ref(mem))),
                  name = maybeRename(port, r, t.ref(mem).field(port))
                )
              )
          }
      }
    case e => e.map(onExpression(_: ir.Expression, r, t))
  }

  /** Rename a statement
    *
    * Instances will update the rename data structure. Memories are treated specially to rename their readers, writers,
    * and readwriters.
    */
  private def onStatement(s: ir.Statement, r: RenameDataStructure, t: ModuleTarget): ir.Statement = s match {
    case decl: ir.IsDeclaration =>
      decl match {
        case decl @ WDefInstance(_, inst, mod, _) =>
          val modx = maybeRename(mod, r, t.circuitTarget.module(mod))
          val instx = doRename(inst, r, t.instOf(inst, mod))
          r.instanceMap(t.ref(inst)) = Right(t.instOf(inst, mod))
          decl.copy(name = instx, module = modx)
        case decl: ir.DefMemory =>
          val namex = doRename(decl.name, r, t.ref(decl.name))
          val tx = t.ref(decl.name)
          r.namespaces(tx) = Namespace(decl.readers ++ decl.writers ++ decl.readwriters)
          r.instanceMap(tx) = Left(tx)
          decl
            .copy(
              name = namex,
              readers = decl.readers.map(_r => doRename(_r, r, tx.field(_r))),
              writers = decl.writers.map(_w => doRename(_w, r, tx.field(_w))),
              readwriters = decl.readwriters.map(_rw => doRename(_rw, r, tx.field(_rw)))
            )
            .map(onExpression(_: ir.Expression, r, t))
        case decl =>
          decl
            .map(doRename(_: String, r, t.ref(decl.name)))
            .map(onExpression(_: ir.Expression, r, t))
      }
    case s =>
      s
        .map(onStatement(_: ir.Statement, r, t))
        .map(onExpression(_: ir.Expression, r, t))
  }

  /** Rename a port */
  private def onPort(p: ir.Port, r: RenameDataStructure, t: ModuleTarget): ir.Port = {
    p.map(doRename(_: String, r, t.ref(p.name)))
  }

  /** Rename a [[DefModule]] and it's internals (ports and statements) to fix keyword collisions and update instance
    * references to respect previous renames
    * @param renames a [[RenameMap]]
    * @param circuit the enclosing [[CircuitName]]
    * @return a [[DefModule]] without keyword conflicts
    */
  private def onModule(m: ir.DefModule, r: RenameDataStructure, t: CircuitTarget): ir.DefModule = m match {
    case _: ir.ExtModule => m
    case ir.Module(_, main, _, _) =>
      val moduleTarget = t.module(m.name)
      r.namespaces(moduleTarget) = Namespace(m)

      /* If top module, use [[maybeRename]]: circuit renaming already recorded a top-module rename if one should happen.
       * Otherwise, use [[doRename]]: compute a new name and record it.
       */
      val onName: String => String = t.circuit match {
        case `main` => maybeRename(_, r, moduleTarget)
        case _      => doRename(_, r, moduleTarget)
      }

      m
        .map(onName)
        .map(onPort(_: ir.Port, r, moduleTarget))
        .map(onStatement(_: ir.Statement, r, moduleTarget))
  }

  /** Manipulate all names in a circuit
    *
    * @param c an input circuit
    * @param renames a rename map that will be updated as names are manipulated
    * @param block a function that returns true if a [[firrtl.annotations.Target Target]] should not be renamed
    * @param allow a function that returns true if a [[firrtl.annotations.Target Target]] should be renamed
    * @return the circuit with manipulated names
    */
  def run(
    c:       ir.Circuit,
    renames: RenameMap,
    block:   Target => Boolean,
    allow:   Target => Boolean
  ): ir.Circuit = {
    val t = CircuitTarget(c.main)

    /* If the circuit is a skip, return the original circuit. Otherwise, walk all the modules and rename them. Rename the
     * circuit if the main module was renamed.
     */
    (block(t), !allow(t)) match {
      case (true, _) =>
        logger.info(s"Circuit '${t.serialize}' is excluded by the 'block' parameter. No renaming will occur.")
        c
      case (false, true) =>
        logger.info(s"Circuit '${t.serialize}' is not included by the 'allow' parameter. No renaming will occur.")
        c
      case _ =>
        val r = new RenameDataStructure(c, renames, block, allow)

        /* Record a rename for the circuit if the top module will be renamed. This allows all the module renames to be
         * aware of the circuit rename when generating their own renames. E.g., this allows renames to be generated
         * that can be resolved in a single step:
         *   ~foo -> FOO
         *   ~foo|bar -> ~FOO|BAR
         * Instead of renames which require multiple steps:
         *   ~foo -> FOO
         *   ~foo|bar -> ~foo|BAR
         */
        val mainx = r.skip(t.module(c.main)) match {
          case true => c.main
          case false =>
            val tx = CircuitTarget(doRename(c.main, r, t))
            logger.info(s"Main module will be renamed. Renaming circuit: '${t.serialize}' -> ['${tx.serialize}']")
            renames.record(t, tx)
            tx.circuit
        }

        /* Rename all modules from leafs to root in one pass while updating a shared rename map. Going from leafs to
         * roots ensures that the rename map is safe for parents to blindly consult. Store this in mapping of old module
         * target to new module to allow the modules to be put in the old order.
         */
        val modulesx: Map[ModuleTarget, ir.DefModule] = InstanceKeyGraph(c).moduleOrder.reverse
          .map(m => t.module(m.name) -> onModule(m, r, t))
          .toMap

        /* Replace the old modules making sure that they are still in the same order */
        c.copy(modules = c.modules.map(m => modulesx(t.module(m.name))), main = mainx)
    }
  }

  /** Return a circuit state with all sensitive names manipulated */
  def execute(state: CircuitState): CircuitState = {

    val block = state.annotations.collect {
      case ManipulateNamesBlocklistAnnotation(targetSeq, t) =>
        t.getObject() match {
          case _: A => targetSeq
          case _ => Nil
        }
    }.flatten.flatten.toSet

    val allow = {
      val allowx = state.annotations.collect {
        case ManipulateNamesAllowlistAnnotation(targetSeq, t) =>
          t.getObject() match {
            case _: A => targetSeq
            case _ => Nil
          }
      }.flatten.flatten

      allowx match {
        case Nil => (a: Target) => true
        case a   => a.toSet
      }
    }

    val renames = RenameMap()
    val circuitx = run(state.circuit, renames, block, allow)

    val annotationsx = state.annotations.flatMap {
      /* Consume blocklist annotations */
      case foo @ ManipulateNamesBlocklistAnnotation(_, t) =>
        t.getObject() match {
          case _: A => None
          case _ => Some(foo)
        }
      /* Convert allowlist annotations to result annotations */
      case foo @ ManipulateNamesAllowlistAnnotation(a, t) =>
        t.getObject() match {
          case _: A =>
            (a, a.map(_.map(renames(_)).flatten)) match {
              case (a, b) => Some(ManipulateNamesAllowlistResultAnnotation(b, t, a))
            }
          case _ => Some(foo)
        }
      case a => Some(a)
    }

    state.copy(circuit = circuitx, annotations = annotationsx, renames = Some(renames))
  }

}
