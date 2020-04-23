// See LICENSE for license details.

package firrtl.analyses

import firrtl.annotations.NoTargetAnnotation
import firrtl.{CircuitState, DependencyAPIMigration, Namespace, Transform}
import firrtl.options.PreservesAll
import firrtl.stage.Forms

case class ModuleNamespaceAnnotation(namespace: Namespace) extends NoTargetAnnotation

/** Create a namespace with this circuit
  *
  * namespace is used by RenameModules to get unique names
  */
class GetNamespace extends Transform with DependencyAPIMigration with PreservesAll[Transform] {
  override def prerequisites = Forms.LowForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.LowEmitters

  def execute(state: CircuitState): CircuitState = {
    val namespace = Namespace(state.circuit)
    state.copy(annotations = new ModuleNamespaceAnnotation(namespace) +: state.annotations)
  }
}
