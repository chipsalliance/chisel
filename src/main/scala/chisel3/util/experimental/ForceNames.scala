// See LICENSE for license details.

package chisel3.util.experimental

import chisel3.experimental.{annotate, ChiselAnnotation, RunFirrtlTransform}
import chisel3.internal.Builder
import firrtl.Mappers._
import firrtl._
import firrtl.annotations._
import firrtl.annotations.TargetToken._
import firrtl.ir._
import firrtl.analyses.InstanceKeyGraph
import firrtl.options.Dependency
import firrtl.passes.{InlineInstances, LowerTypes}
import firrtl.stage.Forms
import firrtl.stage.TransformManager.TransformDependency

import scala.collection.mutable

object forceName {

  /** Force the name of this signal
    *
    * @param signal Signal to name
    * @param name Name to force to
    */
  def apply[T <: chisel3.Element](signal: T, name: String): T = {
    if (!signal.isSynthesizable) Builder.error(s"Using forceName '$name' on non-hardware value $signal")
    annotate(new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = ForceNameAnnotation(signal.toTarget, name)
      override def transformClass: Class[_ <: Transform] = classOf[ForceNamesTransform]
    })
    signal
  }

  /** Force the name of this signal to the name its given during Chisel compilation
    *
    * This will rename after potential renames from other Custom transforms during FIRRTL compilation
    * @param signal Signal to name
    */
  def apply[T <: chisel3.Element](signal: T): T = {
    if (!signal.isSynthesizable) Builder.error(s"Using forceName on non-hardware value $signal")
    annotate(new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = ForceNameAnnotation(signal.toTarget, signal.toTarget.ref)
      override def transformClass: Class[_ <: Transform] = classOf[ForceNamesTransform]
    })
    signal
  }

  /** Force the name of this instance to the name its given during Chisel compilation
    *
    * @param instance Instance to name
    */
  def apply(instance: chisel3.experimental.BaseModule, name: String): Unit = {
    annotate(new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = {
        val t = instance.toAbsoluteTarget
        ForceNameAnnotation(t, name)
      }
      override def transformClass: Class[_ <: Transform] = classOf[ForceNamesTransform]
    })
  }

  /** Force the name of this instance to the name its given during Chisel compilation
    *
    * This will rename after potential renames from other Custom transforms during FIRRTL compilation
    * @param instance Signal to name
    */
  def apply(instance: chisel3.experimental.BaseModule): Unit = {
    annotate(new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = {
        val t = instance.toAbsoluteTarget
        ForceNameAnnotation(t, instance.instanceName)
      }
      override def transformClass: Class[_ <: Transform] = classOf[ForceNamesTransform]
    })
  }
}

/** Links the user-specified name to force to, with the signal/instance in the FIRRTL design
  *
  * @throws CustomTransformException when the signal is renamed to >1 target
  * @param target signal/instance to force the name
  * @param name name to force it to be
  */
case class ForceNameAnnotation(target: IsMember, name: String) extends SingleTargetAnnotation[IsMember] {
  def duplicate(n: IsMember): ForceNameAnnotation = this.copy(target = n, name)

  // Errors if renaming to multiple targets
  override def update(renames: RenameMap): Seq[Annotation] = {
    (target, renames.get(target)) match {
      case (_, None) => List(this)
      case (_: ReferenceTarget, Some(newTargets)) if newTargets.size > 1 =>
        throw CustomTransformException(
          new FirrtlUserException(
            s"Cannot force the name of $target to $name because it is renamed to $newTargets." +
              " Perhaps $target is not a ground type?"
          )
        )
      case (_, Some(newTargets)) => newTargets.map(t => duplicate(t))
    }
  }
}

/** Contains utility functions for [[ForceNamesTransform]]
  *
  * Could (should?) be moved to FIRRTL.
  */
private object ForceNamesTransform {

  /** Returns the [[IsModule]] which is referred to, or if a [[ReferenceTarget]], the enclosing [[IsModule]]
    *
    * @param a signal/instance/module
    * @return referring IsModule
    */
  def referringIsModule(a: IsMember): IsModule = a match {
    case b: ModuleTarget    => b
    case b: InstanceTarget  => b.targetParent
    case b: ReferenceTarget => b.copy(component = Nil).targetParent.asInstanceOf[IsModule]
  }

  /** Returns a function which returns all instance paths to a given IsModule
    *
    * @param graph
    * @return
    */
  def allInstancePaths(graph: InstanceKeyGraph): IsModule => List[List[(Instance, OfModule)]] = {
    val lookup: String => List[List[(Instance, OfModule)]] =
      str =>
        graph
          .findInstancesInHierarchy(str)
          .view
          .map(_.map(_.toTokens).toList)
          .toList
    allInstancePaths(lookup) _
  }

  /** Returns a function which returns all instance paths to a given IsModule
    *
    * @param lookup given a module, return all paths to the module
    * @param target target to get all instance paths to
    * @return
    */
  def allInstancePaths(
    lookup: String => List[List[(Instance, OfModule)]]
  )(target: IsModule
  ): List[List[(Instance, OfModule)]] = {
    target match {
      case ModuleTarget(circuit, module) =>
        if (circuit == module) List(List((Instance(module), OfModule(module))))
        else lookup(module)
      case it: InstanceTarget =>
        val itPath = it.asPath.toList
        if (it.module == it.circuit) List((Instance(it.module), OfModule(it.module)) :: itPath)
        else lookup(it.module).map(_ ++ itPath)
    }
  }

  /** Builds the map of module name to map of old signal/instance name to new signal/instance name
    *
    * @param state CircuitState to operate on
    * @param igraph built instance key graph from state's circuit
    * @return
    */
  def buildForceNameMap(state: CircuitState, igraph: => InstanceKeyGraph): Option[Map[String, Map[String, String]]] = {
    val forceNames = state.annotations.collect { case f: ForceNameAnnotation => f }
    val badNames = mutable.HashSet[ForceNameAnnotation]()
    val allNameMaps = forceNames.groupBy { case f => referringIsModule(f.target) }.mapValues { value =>
      value.flatMap {
        case f @ ForceNameAnnotation(rt: ReferenceTarget, name) if rt.component.nonEmpty =>
          badNames += f
          None
        case ForceNameAnnotation(rt: ReferenceTarget, name) => Some(rt.ref -> name)
        case ForceNameAnnotation(it: InstanceTarget, name) => Some(it.instance -> name)
      }.toMap
    }.toSeq
    val renames: Map[String, Map[String, String]] = {
      val lookup = allInstancePaths(igraph)
      val seen = mutable.Map.empty[List[(Instance, OfModule)], Map[String, String]]
      allNameMaps.foreach {
        case (isModule, nameMap) =>
          // List[Path -> String]
          val result = lookup(isModule).map(_ -> nameMap)
          // Error on collisions
          for ((path, map) <- result) {
            seen.get(path).foreach { old =>
              val msg = s"${prettyPath(path)} annotated with multiple force names! ${old} and ${map}"
              throw new Exception(msg)
            }
            seen(path) = map
          }
        case _ =>
      }
      allNameMaps.map {
        case (isModule, nameMap) => Target.referringModule(isModule).module -> nameMap
      }.toMap
    }
    if (renames.nonEmpty) Some(renames) else None
  }

  /** Returns a nice-looking instance path for error messages */
  def prettyPath(path: Seq[(Instance, OfModule)]): String =
    path.map { case (inst, of) => s"${inst.value}:${of.value}" }.mkString("/")
}

/** Forces the name of marked signals to a certain name
  *   - If there is a conflict in the enclosing module's namespace, throws an exception
  *   - Renames signals of ground types only. If you rename an intermediate module, it will throw an error
  *   - Renames instances as well (as well as port names)
  * Common usages:
  *   - Use to avoid prefixing behavior on specific instances whose enclosing modules are inlined
  */
class ForceNamesTransform extends Transform with DependencyAPIMigration {
  override def optionalPrerequisites:  Seq[TransformDependency] = Seq(Dependency[InlineInstances])
  override def optionalPrerequisiteOf: Seq[TransformDependency] = Forms.LowEmitters
  override def prerequisites:          Seq[TransformDependency] = Seq(Dependency(LowerTypes))
  override def invalidates(a: Transform): Boolean = firrtl.passes.InferTypes == a

  import ForceNamesTransform._

  /** Renames signals/instances in a module
    *
    * @throws FirrtlUserException
    * @param modToNames Maps module name to map of old signal/instance name to new signal/instance name
    * @param renameMap Record renames
    * @param ct Circuit target
    * @param igraph InstanceKeyGraph of parent circuit
    * @param mod Enclosing module of the names to force
    * @return
    */
  private def forceNamesInModule(
    modToNames: Map[String, Map[String, String]],
    renameMap:  RenameMap,
    ct:         CircuitTarget,
    igraph:     InstanceKeyGraph
  )(mod:        DefModule
  ): DefModule = {

    val mt = ct.module(mod.name)
    val instToOfModule = mutable.HashMap[String, String]()
    val names = modToNames.getOrElse(mod.name, Map.empty[String, String])
    // Need to find WRef referring to mems for prefixing
    def onExpr(expr: Expression): Expression = expr match {
      case ref @ Reference(n, _, _, _) if names.contains(n) =>
        ref.copy(name = names(n))
      case sub @ SubField(WRef(i, _, _, _), p, _, _) if instToOfModule.contains(i) =>
        val newsub = modToNames.get(instToOfModule(i)) match {
          case Some(map) if map.contains(p) => sub.copy(name = map(p))
          case _                            => sub
        }
        newsub.map(onExpr)
      case other => other.map(onExpr)
    }
    def onStmt(stmt: Statement): Statement = stmt match {
      // Yes we match on instance name to rename the module
      case inst: DefInstance if names.contains(inst.name) =>
        instToOfModule(inst.name) = inst.module
        val newName = names(inst.name)
        renameMap.record(mt.instOf(inst.name, inst.module), mt.instOf(newName, inst.module))
        inst.copy(name = names(inst.name))
      case inst: DefInstance =>
        instToOfModule(inst.name) = inst.module
        inst
      // Prefix SeqMems because they result in Verilog modules
      case named: IsDeclaration if names.contains(named.name) =>
        renameMap.record(mt.ref(named.name), mt.ref(names(named.name)))
        named.mapString(n => names(n)).map(onStmt).map(onExpr)
      case other => other.map(onStmt).map(onExpr)
    }
    def onPort(port: Port): Port = {
      if (names.contains(port.name)) {
        renameMap.record(mt.ref(port.name), mt.ref(names(port.name)))
        port.copy(name = names(port.name))
      } else port
    }

    val childInstanceHasRename = igraph.getChildInstanceMap(OfModule(mod.name)).exists { o =>
      modToNames.contains(o._2.value)
    }

    if (childInstanceHasRename || modToNames.contains(mod.name)) {
      val ns = Namespace(mod)
      val conflicts = names.values.collect { case k if ns.contains(k) => k }
      if (conflicts.isEmpty) {
        mod.map(onPort).map(onStmt)
      } else {
        throw new FirrtlUserException(
          s"Cannot force the following names in module ${mod.name} because they conflict: ${conflicts.mkString(",")}"
        )
      }
    } else mod
  }

  def execute(state: CircuitState): CircuitState = {
    // Lazy so that it won't be calculated unless buildForceNameMap finds names to force
    lazy val igraph = InstanceKeyGraph(state.circuit)
    buildForceNameMap(state, igraph) match {
      case None =>
        logger.warn("No force names found, skipping...")
        state
      case Some(names) =>
        val renames = RenameMap()
        val cir = state.circuit
        val newCir = cir.mapModule(forceNamesInModule(names, renames, CircuitTarget(cir.main), igraph))
        state.copy(circuit = newCir, renames = Some(renames))
    }
  }
}
