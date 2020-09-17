// SPDX-License-Identifier: Apache-2.0

package firrtl.annotations.transforms

import firrtl.Mappers._
import firrtl.analyses.InstanceKeyGraph
import firrtl.annotations.ModuleTarget
import firrtl.annotations.TargetToken.{fromDefModuleToTargetToken, Instance, OfModule}
import firrtl.annotations.analysis.DuplicationHelper
import firrtl.annotations._
import firrtl.ir._
import firrtl.{AnnotationSeq, CircuitState, DependencyAPIMigration, FirrtlInternalException, RenameMap, Transform}
import firrtl.stage.Forms
import firrtl.transforms.DedupedResult

import scala.collection.mutable

/** Group of targets that should become local targets
  * @param targets
  */
case class ResolvePaths(targets: Seq[CompleteTarget]) extends Annotation {
  override def update(renames: RenameMap): Seq[Annotation] = {
    val newTargets = targets.flatMap(t => renames.get(t).getOrElse(Seq(t)))
    Seq(ResolvePaths(newTargets))
  }
}

/** Holds the mapping from original module to the new, duplicated modules
  * The original module target is unaffected by renaming
  * @param newModules Instance target of what the original module now points to
  * @param originalModule Original module
  */
case class DupedResult(newModules: Set[IsModule], originalModule: ModuleTarget) extends MultiTargetAnnotation {
  override val targets: Seq[Seq[Target]] = Seq(newModules.toSeq)
  override def duplicate(n: Seq[Seq[Target]]): Annotation = {
    n.toList match {
      case Seq(newMods) => DupedResult(newMods.collect { case x: IsModule => x }.toSet, originalModule)
      case _            => DupedResult(Set.empty, originalModule)
    }
  }
}

case class NoSuchTargetException(message: String) extends FirrtlInternalException(message)

object EliminateTargetPaths {

  def renameModules(c: Circuit, toRename: Map[String, String], renameMap: RenameMap): Circuit = {
    val ct = CircuitTarget(c.main)
    val cx = if (toRename.contains(c.main)) {
      renameMap.record(ct, CircuitTarget(toRename(c.main)))
      c.copy(main = toRename(c.main))
    } else {
      c
    }
    def onMod(m: DefModule): DefModule = {
      m.map(onStmt) match {
        case e: ExtModule if toRename.contains(e.name) =>
          renameMap.record(ct.module(e.name), ct.module(toRename(e.name)))
          e.copy(name = toRename(e.name))
        case e: Module if toRename.contains(e.name) =>
          renameMap.record(ct.module(e.name), ct.module(toRename(e.name)))
          e.copy(name = toRename(e.name))
        case o => o
      }
    }
    def onStmt(s: Statement): Statement = s.map(onStmt) match {
      case w @ DefInstance(info, name, module, _) if toRename.contains(module) => w.copy(module = toRename(module))
      case other                                                               => other
    }
    cx.map(onMod)
  }

  def reorderModules(c: Circuit, toReorder: Map[String, Double]): Circuit = {
    val newOrderMap = c.modules.zipWithIndex.map {
      case (m, _) if toReorder.contains(m.name) => m.name -> toReorder(m.name)
      case (m, i) if c.modules.size > 1         => m.name -> i.toDouble / (c.modules.size - 1)
      case (m, _)                               => m.name -> 1.0
    }.toMap

    val newOrder = c.modules.sortBy { m => newOrderMap(m.name) }

    c.copy(modules = newOrder)
  }

}

/** For a set of non-local targets, modify the instance/module hierarchy of the circuit such that
  * the paths in each non-local target can be removed
  *
  * In other words, if targeting a specific instance of a module, duplicate that module with a unique name
  * and instantiate the new module instead.
  *
  * Consumes [[ResolvePaths]]
  *
  * E.g. for non-local target A/b:B/c:C/d, rename the following
  * A/b:B/c:C/d -> C_/d
  * A/b:B/c:C -> B_/c:C_
  * A/b:B -> A/b:B_
  * B/x -> (B/x, B_/x) // where x is any reference in B
  * C/x -> (C/x, C_/x) // where x is any reference in C
  */
class EliminateTargetPaths extends Transform with DependencyAPIMigration {
  import EliminateTargetPaths._

  override def prerequisites = Forms.MinimalHighForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Transform) = false

  /** Replaces old ofModules with new ofModules by calling dupMap methods
    * Updates oldUsedOfModules, newUsedOfModules
    * @param originalModule Original name of this module
    * @param newModule New name of this module
    * @param s
    * @return
    */
  private def onStmt(dupMap: DuplicationHelper)(originalModule: String, newModule: String)(s: Statement): Statement =
    s match {
      case d @ DefInstance(_, name, module, _) =>
        val ofModule = dupMap.getNewOfModule(originalModule, newModule, Instance(name), OfModule(module)).value
        d.copy(module = ofModule)
      case other => other.map(onStmt(dupMap)(originalModule, newModule))
    }

  /** Returns a modified circuit and [[RenameMap]] containing the associated target remapping
    * @param cir
    * @param targets
    * @return
    */
  def run(cir: Circuit, targets: Seq[IsMember], iGraph: InstanceKeyGraph): (Circuit, RenameMap, AnnotationSeq) = {

    val dupMap = DuplicationHelper(cir.modules.map(_.name).toSet)

    // For each target, record its path and calculate the necessary modifications to circuit
    targets.foreach { t => dupMap.expandHierarchy(t) }

    // Contains new list of module declarations
    val duplicatedModuleList = mutable.ArrayBuffer[DefModule]()

    // Foreach module, calculate the unique names of its duplicates
    // Then, update the ofModules of instances that it encapsulates

    val ct = CircuitTarget(cir.main)
    val annos = cir.modules.map { m =>
      val newNames = dupMap.getDuplicates(m.name)
      newNames.foreach { newName =>
        val newM = m match {
          case e: ExtModule => e.copy(name = newName)
          case o: Module =>
            o.copy(name = newName, body = onStmt(dupMap)(m.name, newName)(o.body))
        }
        duplicatedModuleList += newM
      }
      DupedResult(newNames.map(ct.module), ct.module(m.name))
    }

    val finalModuleList = duplicatedModuleList
    lazy val finalModuleSet = finalModuleList.map { case a: DefModule => a.name }.toSet

    // Records how targets have been renamed
    val renameMap = RenameMap()

    /* Foreach target, calculate the pathless version and only rename targets that are instantiated. Additionally, rename
     * module targets
     */
    def addRecord(old: IsMember, newPathless: IsMember): Unit = old match {
      case x: ModuleTarget =>
        renameMap.record(x, newPathless)
      case x: IsComponent if x.path.isEmpty =>
        renameMap.record(x, newPathless)
      case x: IsComponent =>
        renameMap.record(x, newPathless)
        addRecord(x.stripHierarchy(1), newPathless)
    }
    val duplicatedParents = mutable.Set[OfModule]()
    targets.foreach { t =>
      val newTs = dupMap.makePathless(t)
      val path = t.asPath
      if (path.nonEmpty) { duplicatedParents += path(0)._2 }
      newTs.toList match {
        case Seq(pathless) =>
          val mt = Target.referringModule(pathless)
          addRecord(t, pathless)
          renameMap.record(Target.referringModule(t), mt)
        case _ =>
      }
    }

    def addSelfRecord(mod: IsModule): Unit = mod match {
      case m: ModuleTarget =>
      case i: InstanceTarget if renameMap.underlying.contains(i) =>
      case i: InstanceTarget =>
        renameMap.record(i, i)
        addSelfRecord(i.stripHierarchy(1))
    }
    val topMod = ModuleTarget(cir.main, cir.main)
    duplicatedParents.foreach { parent =>
      val paths = iGraph.findInstancesInHierarchy(parent.value)
      val newTargets = paths.map { path =>
        path.tail.foldLeft(topMod: IsModule) {
          case (mod, wDefInst) =>
            mod.instOf(wDefInst.name, wDefInst.module)
        }
      }
      newTargets.foreach(addSelfRecord(_))
    }

    // Return modified circuit and associated renameMap
    (cir.copy(modules = finalModuleList.toSeq), renameMap, annos)
  }

  override def execute(state: CircuitState): CircuitState = {
    val moduleNames = state.circuit.modules.map(_.name).toSet

    val (remainingAnnotations, targetsToEliminate, previouslyDeduped) =
      state.annotations.foldLeft(
        (Vector.empty[Annotation], Seq.empty[CompleteTarget], Map.empty[IsModule, (ModuleTarget, Double)])
      ) {
        case ((remainingAnnos, targets, dedupedResult), anno) =>
          anno match {
            case ResolvePaths(ts) =>
              (remainingAnnos, ts ++ targets, dedupedResult)
            case DedupedResult(orig, dups, idx) if dups.nonEmpty =>
              (remainingAnnos, targets, dedupedResult ++ dups.map(_ -> (orig, idx)).toMap)
            case other =>
              (remainingAnnos :+ other, targets, dedupedResult)
          }
      }

    // Collect targets that are not local
    val targets = targetsToEliminate.collect { case x: IsMember => x }

    // Check validity of paths in targets
    val iGraph = InstanceKeyGraph(state.circuit)
    val instanceOfModules = iGraph.getChildInstances.map { case (k, v) => k -> v.map(_.toTokens) }.toMap
    val targetsWithInvalidPaths = mutable.ArrayBuffer[IsMember]()
    targets.foreach { t =>
      val path = t match {
        case _: ModuleTarget    => Nil
        case i: InstanceTarget  => i.asPath
        case r: ReferenceTarget => r.path
      }
      path.foldLeft(t.module) {
        case (module, (inst: Instance, of: OfModule)) =>
          val childrenOpt = instanceOfModules.get(module)
          if (childrenOpt.isEmpty || !childrenOpt.get.contains((inst, of))) {
            targetsWithInvalidPaths += t
          }
          of.value
      }
    }
    if (targetsWithInvalidPaths.nonEmpty) {
      val string = targetsWithInvalidPaths.mkString(",")
      throw NoSuchTargetException(s"""Some targets have illegal paths that cannot be resolved/eliminated: $string""")
    }

    /* get rid of path prefixes of modules with only one instance so we don't rename them
     * If instance targeted is in fact the only instance of a module, then it should not be renamed
     * E.g. if Eliminate Target Paths on ~Top|Top/foo:Foo, but that is the only instance of Foo, then should return
     *   ~Top|Top/foo:Foo, not ~Top|Top/foo:Foo___Top_foo
     */
    val isSingleInstMod: String => Boolean = {
      val cache = mutable.Map.empty[String, Boolean]
      mod => cache.getOrElseUpdate(mod, iGraph.findInstancesInHierarchy(mod).size == 1)
    }
    val firstRenameMap = RenameMap()
    val nonSingletonTargets = targets.foldRight(Seq.empty[IsMember]) {
      case (t: IsComponent, acc) if t.asPath.nonEmpty =>
        val origPath = t.asPath
        val (singletonPrefix, rest) = origPath.partition {
          case (_, OfModule(mod)) =>
            isSingleInstMod(mod)
        }

        if (singletonPrefix.size > 0) {
          val module = singletonPrefix.last._2.value
          val circuit = t.circuit
          val newIsModule =
            if (rest.isEmpty) {
              ModuleTarget(circuit, module)
            } else {
              val (Instance(inst), OfModule(ofMod)) = rest.last
              val path = rest.dropRight(1)
              InstanceTarget(circuit, module, path, inst, ofMod)
            }
          val newTarget = t match {
            case r: ReferenceTarget => r.setPathTarget(newIsModule)
            case i: InstanceTarget  => newIsModule
          }
          firstRenameMap.record(t, Seq(newTarget))
          newTarget +: acc
        } else {
          t +: acc
        }
      case (t, acc) => t +: acc
    }

    val (newCircuit, nextRenameMap, newAnnos) = run(state.circuit, nonSingletonTargets, iGraph)

    val renameMap =
      if (firstRenameMap.hasChanges) {
        firstRenameMap.andThen(nextRenameMap)
      } else {
        nextRenameMap
      }

    val iGraphx = InstanceKeyGraph(newCircuit)
    val newlyUnreachableModules = iGraphx.unreachableModules.toSet.diff(iGraph.unreachableModules.toSet)

    val newCircuitGC = {
      val modulesx = newCircuit.modules.flatMap {
        case dead if newlyUnreachableModules(dead.OfModule) => None
        case live =>
          val m = CircuitTarget(newCircuit.main).module(live.name)
          renameMap.get(m).foreach(_ => renameMap.record(m, m))
          Some(live)
      }
      newCircuit.copy(modules = modulesx)
    }

    val renamedModuleMap = RenameMap()

    // If previous instance target mapped to a single previously deduped module, return original name
    // E.g. if previously ~Top|Top/foo:Foo was deduped to ~Top|Top/foo:Bar, then
    //  Eliminate target paths on ~Top|Top/foo:Bar should rename to ~Top|Top/foo:Foo, not
    //  ~Top|Top/foo:Bar___Top_foo
    val newModuleNameMapping = previouslyDeduped.flatMap {
      case (current: IsModule, (orig: ModuleTarget, idx)) =>
        renameMap.get(current).collect { case Seq(ModuleTarget(_, m)) => m -> orig.name }
    }

    val renamedCircuit = renameModules(newCircuitGC, newModuleNameMapping, renamedModuleMap)

    val reorderedCircuit = reorderModules(
      renamedCircuit,
      previouslyDeduped.map {
        case (current: IsModule, (orig: ModuleTarget, idx)) =>
          orig.name -> idx
      }
    )

    state.copy(
      circuit = reorderedCircuit,
      renames = Some(renameMap.andThen(renamedModuleMap)),
      annotations = remainingAnnotations ++ newAnnos
    )
  }
}
