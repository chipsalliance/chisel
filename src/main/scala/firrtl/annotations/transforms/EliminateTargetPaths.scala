// See LICENSE for license details.

package firrtl.annotations.transforms

import firrtl.Mappers._
import firrtl.analyses.InstanceGraph
import firrtl.annotations.ModuleTarget
import firrtl.annotations.TargetToken.{Instance, OfModule, fromDefModuleToTargetToken}
import firrtl.annotations.analysis.DuplicationHelper
import firrtl.annotations._
import firrtl.ir._
import firrtl.{CircuitForm, CircuitState, FirrtlInternalException, HighForm, RenameMap, Transform, WDefInstance}

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

case class NoSuchTargetException(message: String) extends FirrtlInternalException(message)

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
class EliminateTargetPaths extends Transform {

  def inputForm: CircuitForm = HighForm

  def outputForm: CircuitForm = HighForm

  /** Replaces old ofModules with new ofModules by calling dupMap methods
    * Updates oldUsedOfModules, newUsedOfModules
    * @param originalModule Original name of this module
    * @param newModule New name of this module
    * @param s
    * @return
    */
  private def onStmt(dupMap: DuplicationHelper)
                    (originalModule: String, newModule: String)
                    (s: Statement): Statement = s match {
    case d@DefInstance(_, name, module) =>
      val ofModule = dupMap.getNewOfModule(originalModule, newModule, Instance(name), OfModule(module)).value
      d.copy(module = ofModule)
    case d@WDefInstance(_, name, module, _) =>
      val ofModule = dupMap.getNewOfModule(originalModule, newModule, Instance(name), OfModule(module)).value
      d.copy(module = ofModule)
    case other => other map onStmt(dupMap)(originalModule, newModule)
  }

  /** Returns a modified circuit and [[RenameMap]] containing the associated target remapping
    * @param cir
    * @param targets
    * @return
    */
  def run(cir: Circuit, targets: Seq[IsMember]): (Circuit, RenameMap) = {

    val dupMap = DuplicationHelper(cir.modules.map(_.name).toSet)

    // For each target, record its path and calculate the necessary modifications to circuit
    targets.foreach { t => dupMap.expandHierarchy(t) }

    // Contains new list of module declarations
    val duplicatedModuleList = mutable.ArrayBuffer[DefModule]()

    // Foreach module, calculate the unique names of its duplicates
    // Then, update the ofModules of instances that it encapsulates
    cir.modules.foreach { m =>
      dupMap.getDuplicates(m.name).foreach { newName =>
        val newM = m match {
          case e: ExtModule => e.copy(name = newName)
          case o: Module =>
            o.copy(name = newName, body = onStmt(dupMap)(m.name, newName)(o.body))
        }
        duplicatedModuleList += newM
      }
    }

    val finalModuleList = duplicatedModuleList
    lazy val finalModuleSet = finalModuleList.map{ case a: DefModule => a.name }.toSet

    // Records how targets have been renamed
    val renameMap = RenameMap()

    /* Foreach target, calculate the pathless version and only rename targets that are instantiated. Additionally, rename
     * module targets
     */
    targets.foreach { t =>
      val newTsx = dupMap.makePathless(t)
      val newTs = newTsx
      if(newTs.nonEmpty) {
        renameMap.record(t, newTs)
        val m = Target.referringModule(t)
        val duplicatedModules = newTs.map(Target.referringModule)
        val oldModule: Option[ModuleTarget] = m match {
          case a: ModuleTarget if finalModuleSet(a.module) => Some(a)
          case _                                           => None
        }
        renameMap.record(m, (duplicatedModules).distinct)
      }
    }

    // Return modified circuit and associated renameMap
    (cir.copy(modules = finalModuleList), renameMap)
  }

  override protected def execute(state: CircuitState): CircuitState = {

    val (annotations, annotationsx) = state.annotations.partition{
      case a: ResolvePaths => true
      case _               => false
    }

    // Collect targets that are not local
    val targets = annotations.map(_.asInstanceOf[ResolvePaths]).flatMap(_.targets.collect { case x: IsMember => x })

    // Check validity of paths in targets
    val iGraph = new InstanceGraph(state.circuit)
    val instanceOfModules = iGraph.getChildrenInstanceOfModule
    val targetsWithInvalidPaths = mutable.ArrayBuffer[IsMember]()
    targets.foreach { t =>
      val path = t match {
        case m: ModuleTarget => Nil
        case i: InstanceTarget => i.asPath
        case r: ReferenceTarget => r.path
      }
      path.foldLeft(t.module) { case (module, (inst: Instance, of: OfModule)) =>
        val childrenOpt = instanceOfModules.get(module)
        if(childrenOpt.isEmpty || !childrenOpt.get.contains((inst, of))) {
          targetsWithInvalidPaths += t
        }
        of.value
      }
    }
    if(targetsWithInvalidPaths.nonEmpty) {
      val string = targetsWithInvalidPaths.mkString(",")
      throw NoSuchTargetException(s"""Some targets have illegal paths that cannot be resolved/eliminated: $string""")
    }

    val (newCircuit, renameMap) = run(state.circuit, targets)

    val iGraphx = new InstanceGraph(newCircuit)
    val newlyUnreachableModules = iGraphx.unreachableModules diff iGraph.unreachableModules

    val newCircuitGC = {
      val modulesx = newCircuit.modules.flatMap{
        case dead if newlyUnreachableModules(dead.OfModule) => None
        case live =>
          val m = CircuitTarget(newCircuit.main).module(live.name)
          renameMap.get(m).foreach(_ => renameMap.record(m, m))
          Some(live)
      }
      newCircuit.copy(modules = modulesx)
    }

    logger.info("Renames:")
    logger.info(renameMap.serialize)

    state.copy(circuit = newCircuitGC, renames = Some(renameMap), annotations = annotationsx)
  }
}
