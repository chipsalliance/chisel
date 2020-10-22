// See LICENSE for license details.

package chisel3.util.experimental

import chisel3.experimental.{annotate, ChiselAnnotation}
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
  def apply(signal: => chisel3.Data, name: String): Unit = {
    annotate(new ChiselAnnotation { def toFirrtl =
      ForceNameAnnotation(signal.toTarget, name)
    })
  }
  def apply(instance: chisel3.experimental.BaseModule, name: String): Unit = {
    annotate(new ChiselAnnotation {
      def toFirrtl = {
        val t = instance.toAbsoluteTarget
        ForceNameAnnotation(t, name)
      }
    })
  }
  def apply(instance: chisel3.experimental.BaseModule): Unit = {
    annotate(new ChiselAnnotation {
      def toFirrtl = {
        val t = instance.toAbsoluteTarget
        ForceNameAnnotation(t, instance.instanceName)
      }
    })
  }
}

case class ForceNameAnnotation(target: IsMember, name: String)
    extends SingleTargetAnnotation[IsMember] {
  def duplicate(n: IsMember): ForceNameAnnotation = this.copy(target = n, name)
}


object ForceNamesTransform {
  def referringIsModule(a: IsMember): IsModule = a match {
    case b: ModuleTarget    => b
    case b: InstanceTarget  => b.targetParent
    case b: ReferenceTarget => b.copy(component = Nil).targetParent.asInstanceOf[IsModule]
  }
  def allInstancePaths(graph: InstanceKeyGraph): IsModule => List[List[(Instance, OfModule)]] = {
    val lookup: String => List[List[(Instance, OfModule)]] =
      str => graph.findInstancesInHierarchy(str)
        .view
        .map(_.map(_.toTokens).toList)
        .toList
    allInstancePaths(lookup) _
  }
  // Use Lists because we want to recurse on inner List (head :: tail)-style
  def allInstancePaths(lookup: String => List[List[(Instance, OfModule)]])
                      (target: IsModule): List[List[(Instance, OfModule)]] = {
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


  def buildForceNameMap(state: CircuitState,
                     igraph: => InstanceKeyGraph
                    ): Option[Map[String, Map[String, String]]] = {
    val forceNames = state.annotations.collect { case f: ForceNameAnnotation => f }
    val badNames = mutable.HashSet[ForceNameAnnotation]()
    val allNameMaps = forceNames.groupBy { case f => referringIsModule(f.target) }.mapValues { value =>
      value.flatMap {
        case f@ForceNameAnnotation(rt: ReferenceTarget, name) if rt.component.nonEmpty =>
          badNames += f
          None
        case ForceNameAnnotation(rt: ReferenceTarget, name) => Some(rt.ref -> name)
        case ForceNameAnnotation(it: InstanceTarget, name) => Some(it.instance -> name)
      }.toMap
    }
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
      }
    }
    if (renames.nonEmpty) { Some(renames) } else None
  }

  def prettyPath(path: Seq[(Instance, OfModule)]): String =
    path.map { case (inst, of) => s"${inst.value}:${of.value}" }.mkString("/")
}

/** Prefixes all Modules (and SeqMem instances) with the prefix in a [[PrefixModulesAnnotation]]
  *
  * ReplSeqMems uses the name of the instance of the DefMemory, so we need to prefix that
  *
  * @note SeqMems are [[DefMemory]]s with readlatency = writelatency = 1
  * @note Does NOT prefix ExtModules
  * TODO
  *  - Improve performance when prefixed modules don't have instances nor mems
  */
class ForceNamesTransform extends Transform with DependencyAPIMigration {
  // Must run before ExtractSeqMems and after ReplSeqMems
  override def optionalPrerequisites: Seq[TransformDependency] = Seq(Dependency[InlineInstances])
  override def optionalPrerequisiteOf: Seq[TransformDependency] = Forms.LowEmitters

  override def prerequisites: Seq[TransformDependency] = Seq(Dependency(LowerTypes))

  override def invalidates(a: Transform): Boolean = firrtl.passes.InferTypes == a

  import ForceNamesTransform._

  private def forceNamesInModule(modToNames: Map[String, Map[String, String]], renameMap: RenameMap, ct: CircuitTarget, igraph: InstanceKeyGraph)
                                (mod: DefModule): DefModule = {

    val mt = ct.module(mod.name)
    val instToOfModule = mutable.HashMap[String, String]()
    val names = modToNames.getOrElse(mod.name, Map.empty[String, String])
    // Need to find WRef referring to mems for prefixing
    def onExpr(expr: Expression): Expression = expr match {
      case wref @ WRef(n, _,_,_) if names.contains(n) =>
        wref.copy(name = names(n))
      case sub @ SubField(WRef(i, _, _, _), p,_,_) if instToOfModule.contains(i) =>
        val newsub = modToNames.get(instToOfModule(i)) match {
          case Some(map) if map.contains(p) => sub.copy(name = map(p))
          case _ => sub
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
      case error @ (_: WDefInstanceConnector) =>
        throw new Exception(s"Unexpected instance object $error")
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

    val childInstanceHasRename = igraph.getChildInstanceMap(OfModule(mod.name)).valuesIterator.collectFirst {
        case o if modToNames.contains(o.value) => o
      }.isDefined

    if(childInstanceHasRename || modToNames.contains(mod.name)) {
      mod.map(onPort).map(onStmt)
    } else mod
  }

  def execute(state: CircuitState): CircuitState = {
    // Lazy so that it won't be calculated unless buildPrefixMap finds requisite annotations
    lazy val igraph = InstanceKeyGraph(state.circuit)
    buildForceNameMap(state, igraph) match {
      case None =>
        logger.warn("No prefixes found, skipping...")
        state
      case Some(names) =>
        val renames = RenameMap()
        val cir = state.circuit
        val newCir = cir.mapModule(forceNamesInModule(names, renames, CircuitTarget(cir.main), igraph))
        state.copy(circuit = newCir, renames = Some(renames))
    }
  }
}
