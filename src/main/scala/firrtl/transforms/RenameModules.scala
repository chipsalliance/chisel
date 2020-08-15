// See LICENSE for license details.

package firrtl.transforms

import firrtl.analyses.{InstanceKeyGraph, ModuleNamespaceAnnotation}
import firrtl.ir._
import firrtl._
import firrtl.stage.Forms

import scala.collection.mutable

/** Rename Modules
  *
  * using namespace created by [[analyses.GetNamespace]], create unique names for modules
  */
class RenameModules extends Transform with DependencyAPIMigration {

  override def prerequisites = Forms.LowForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.LowEmitters
  override def invalidates(a: Transform) = false

  def collectNameMapping(namespace: Namespace, moduleNameMap: mutable.HashMap[String, String])(mod: DefModule): Unit = {
    val newName = namespace.newName(mod.name)
    moduleNameMap.put(mod.name, newName)
  }

  def onStmt(moduleNameMap: mutable.HashMap[String, String])(stmt: Statement): Statement = stmt match {
    case inst: WDefInstance if moduleNameMap.contains(inst.module) => inst.copy(module = moduleNameMap(inst.module))
    case other => other.mapStmt(onStmt(moduleNameMap))
  }

  def execute(state: CircuitState): CircuitState = {
    val namespace = state.annotations.collectFirst {
      case m: ModuleNamespaceAnnotation => m
    }.map(_.namespace)

    if (namespace.isEmpty) {
      logger.warn("Skipping Rename Modules")
      state
    } else {
      val moduleOrder = InstanceKeyGraph(state.circuit).moduleOrder.reverse
      val nameMappings = new mutable.HashMap[String, String]()
      moduleOrder.foreach(collectNameMapping(namespace.get, nameMappings))

      val modulesx = state.circuit.modules.map {
        case mod: Module    => mod.mapStmt(onStmt(nameMappings)).mapString(nameMappings)
        case ext: ExtModule => ext
      }

      state.copy(circuit = state.circuit.copy(modules = modulesx, main = nameMappings(state.circuit.main)))
    }
  }
}
