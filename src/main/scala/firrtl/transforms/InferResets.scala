// See LICENSE for license details.

package firrtl.transforms

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.traversals.Foreachers._
import firrtl.annotations.{ReferenceTarget, TargetToken}
import firrtl.Utils.toTarget
import firrtl.passes.{Pass, PassException, Errors, InferTypes}

import scala.collection.mutable
import scala.util.Try

object InferResets {
  final class DifferingDriverTypesException private (msg: String) extends PassException(msg)
  object DifferingDriverTypesException {
    def apply(target: ReferenceTarget, tpes: Seq[(Type, Seq[TypeDriver])]): DifferingDriverTypesException = {
      val xs = tpes.map { case (t, ds) => s"${ds.map(_.target().serialize).mkString(", ")} of type ${t.serialize}" }
      val msg = s"${target.serialize} driven with multiple types!" + xs.mkString("\n  ", "\n  ", "")
      new DifferingDriverTypesException(msg)
    }
  }

  /** Type hierarchy to represent the type of the thing driving a [[ResetType]] */
  private sealed trait ResetDriver
  // When a [[ResetType]] is driven by another ResetType, we track the target so that we can infer
  //   the same type as the driver
  private case class TargetDriver(target: ReferenceTarget) extends ResetDriver {
    override def toString: String = s"TargetDriver(${target.serialize})"
  }
  // When a [[ResetType]] is driven by something of type Bool or AsyncResetType, we keep track of it
  //   as a constraint on the type we should infer to be
  // We keep the target around (lazily) so that we can report errors
  private case class TypeDriver(tpe: Type, target: () => ReferenceTarget) extends ResetDriver {
    override def toString: String = s"TypeDriver(${tpe.serialize}, $target)"
  }


  // Type hierarchy representing the path to a leaf type in an aggregate type structure
  // Used by this [[InferResets]] to pinpoint instances of [[ResetType]] and their inferred type
  private sealed trait TypeTree
  private case class BundleTree(fields: Map[String, TypeTree]) extends TypeTree
  private case class VectorTree(subType: TypeTree) extends TypeTree
  // TODO ensure is only AsyncResetType or BoolType
  private case class GroundTree(tpe: Type) extends TypeTree

  private object TypeTree {
    // Given groups of [[TargetToken]]s and Types corresponding to them, construct a [[TypeTree]]
    // that allows us to lookup the type of each leaf node in the aggregate structure
    // TODO make return Try[TypeTree]
    def fromTokens(tokens: (Seq[TargetToken], Type)*): TypeTree = tokens match {
      case Seq((Seq(), tpe)) => GroundTree(tpe)
      // VectorTree
      case (TargetToken.Index(_) +: _, _) +: _ =>
        // Vectors must all have the same type, so we only process Index 0
        // If the subtype is an aggregate, there can be multiple of each index
        val ts = tokens.collect { case (TargetToken.Index(0) +: tail, tpe) => (tail, tpe) }
        VectorTree(fromTokens(ts:_*))
      // BundleTree
      case (TargetToken.Field(_) +: _, _) +: _ =>
        val fields =
          tokens.groupBy { case (TargetToken.Field(n) +: t, _) => n }
                .mapValues { ts =>
                  fromTokens(ts.map { case (_ +: t, tpe) => (t, tpe) }:_*)
                }
        BundleTree(fields)
    }
  }
}

/** Infers the concrete type of [[Reset]]s by their connections
  * This is a global inference because ports can be of type [[Reset]]
  * @note This transform should be run before [[DedupModules]] so that similar Modules from
  *   generator languages like Chisel can infer differently
  */
// TODO should we error if a DefMemory is of type AsyncReset? In CheckTypes?
class InferResets extends Transform {
  def inputForm: CircuitForm = HighForm
  def outputForm: CircuitForm = HighForm

  import InferResets._

  // Collect all drivers for circuit elements of type ResetType
  private def analyze(c: Circuit): Map[ReferenceTarget, List[ResetDriver]] = {
    type DriverMap = mutable.HashMap[ReferenceTarget, mutable.ListBuffer[ResetDriver]]
    def onMod(mod: DefModule): DriverMap = {
      val instMap = mutable.Map[String, String]()
      // We need to convert submodule port targets into targets on the Module port itself
      def makeTarget(expr: Expression): ReferenceTarget = {
        val target = toTarget(c.main, mod.name)(expr)
        Utils.kind(expr) match {
          case InstanceKind =>
            val mod = instMap(target.ref)
            val port = target.component.head match {
              case TargetToken.Field(name) => name
              case bad => Utils.throwInternalError(s"Unexpected token $bad")
            }
            target.copy(module = mod, ref = port, component = target.component.tail)
          case _ => target
        }
      }
      def onStmt(map: DriverMap)(stmt: Statement): Unit = {
        // Mark driver of a ResetType leaf
        def markResetDriver(lhs: Expression, rhs: Expression): Unit = {
          val lflip = Utils.to_flip(Utils.gender(lhs))
          if ((lflip == Default && lhs.tpe == ResetType) ||
              (lflip == Flip    && rhs.tpe == ResetType)) {
            val (loc, exp) = lflip match {
              case Default => (lhs, rhs)
              case Flip    => (rhs, lhs)
            }
            val target = makeTarget(loc)
            val driver = exp.tpe match {
              case ResetType => TargetDriver(makeTarget(exp))
              case tpe       => TypeDriver(tpe, () => makeTarget(exp))
            }
            map.getOrElseUpdate(target, mutable.ListBuffer()) += driver
          }
        }
        stmt match {
          // TODO
          //  - Each connect duplicates a bunch of code from ExpandConnects, could be cleaner
          //  - The full create_exps duplication is inefficient, there has to be a better way
          case Connect(_, lhs, rhs) =>
            val locs = Utils.create_exps(lhs)
            val exps = Utils.create_exps(rhs)
            for ((loc, exp) <- locs.zip(exps)) {
              markResetDriver(loc, exp)
            }
          case PartialConnect(_, lhs, rhs) =>
            val points = Utils.get_valid_points(lhs.tpe, rhs.tpe, Default, Default)
            val locs = Utils.create_exps(lhs)
            val exps = Utils.create_exps(rhs)
            for ((i, j) <- points) {
              markResetDriver(locs(i), exps(j))
            }
          case WDefInstance(_, inst, module, _) =>
            instMap += (inst -> module)
          case Conditionally(_, _, con, alt) =>
            val conMap = new DriverMap
            val altMap = new DriverMap
            onStmt(conMap)(con)
            onStmt(altMap)(alt)
            // Default to outerscope if not found in alt
            val altLookup = altMap.orElse(map).lift
            for (key <- conMap.keys ++ altMap.keys) {
              val ds = map.getOrElseUpdate(key, mutable.ListBuffer())
              conMap.get(key).foreach(ds ++= _)
              altLookup(key).foreach(ds ++= _)
            }
          case other => other.foreach(onStmt(map))
        }
      }
      val types = new DriverMap
      mod.foreach(onStmt(types))
      types
    }
    c.modules.foldLeft(Map[ReferenceTarget, List[ResetDriver]]()) {
      case (map, mod) => map ++ onMod(mod).mapValues(_.toList)
    }
  }

  // Determine the type driving a given ResetType
  private def resolve(map: Map[ReferenceTarget, List[ResetDriver]]): Try[Map[ReferenceTarget, Type]] = {
    val res = mutable.Map[ReferenceTarget, Type]()
    val errors = new Errors
    def rec(target: ReferenceTarget): Type = {
      val drivers = map(target)
      res.getOrElseUpdate(target, {
        val tpes = drivers.map {
          case TargetDriver(t) => TypeDriver(rec(t), () => t)
          case td: TypeDriver => td
        }.groupBy(_.tpe)
        if (tpes.keys.size != 1) {
          // Multiple types of driver!
          errors.append(DifferingDriverTypesException(target, tpes.toSeq))
        }
        tpes.keys.head
      })
    }
    for ((target, _) <- map) {
      rec(target)
    }
    Try {
      errors.trigger()
      res.toMap
    }
  }

  private def fixupType(tpe: Type, tree: TypeTree): Type = (tpe, tree) match {
    case (BundleType(fields), BundleTree(map)) =>
      val fieldsx =
        fields.map(f => map.get(f.name) match {
          case Some(t) => f.copy(tpe = fixupType(f.tpe, t))
          case None => f
        })
      BundleType(fieldsx)
    case (VectorType(vtpe, size), VectorTree(t)) =>
      VectorType(fixupType(vtpe, t), size)
    case (_, GroundTree(t)) => t
    case x => throw new Exception(s"Error! Unexpected pair $x")
  }

  // Assumes all ReferenceTargets are in the same module
  private def makeDeclMap(map: Map[ReferenceTarget, Type]): Map[String, TypeTree] =
    map.groupBy(_._1.ref).mapValues { ts =>
      TypeTree.fromTokens(ts.toSeq.map { case (target, tpe) => (target.component, tpe) }:_*)
    }

  private def implPort(map: Map[String, TypeTree])(port: Port): Port =
    map.get(port.name)
       .map(tree => port.copy(tpe = fixupType(port.tpe, tree)))
       .getOrElse(port)
  private def implStmt(map: Map[String, TypeTree])(stmt: Statement): Statement =
    stmt.map(implStmt(map)) match {
      case decl: IsDeclaration if map.contains(decl.name) =>
        val tree = map(decl.name)
        decl match {
          case reg: DefRegister => reg.copy(tpe = fixupType(reg.tpe, tree))
          case wire: DefWire => wire.copy(tpe = fixupType(wire.tpe, tree))
          // TODO Can this really happen?
          case mem: DefMemory => mem.copy(dataType = fixupType(mem.dataType, tree))
          case other => other
        }
      case other => other
    }

  private def implement(c: Circuit, map: Map[ReferenceTarget, Type]): Circuit = {
    val modMaps = map.groupBy(_._1.module)
    def onMod(mod: DefModule): DefModule = {
      modMaps.get(mod.name).map { tmap =>
        val declMap = makeDeclMap(tmap)
        mod.map(implPort(declMap)).map(implStmt(declMap))
      }.getOrElse(mod)
    }
    c.map(onMod)
  }

  private def fixupPasses: Seq[Pass] = Seq(
    InferTypes
  )

  def execute(state: CircuitState): CircuitState = {
    val c = state.circuit
    val analysis = analyze(c)
    val inferred = resolve(analysis)
    val result = inferred.map(m => implement(c, m)).get
    val fixedup = fixupPasses.foldLeft(result)((c, p) => p.run(c))
    state.copy(circuit = fixedup)
  }
}
