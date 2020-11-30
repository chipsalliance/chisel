// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms

import firrtl.{CircuitState, DependencyAPIMigration, Transform}
import firrtl.analyses.InstanceKeyGraph
import firrtl.options.Dependency
import firrtl.stage.Forms

/** Return a circuit where all modules (and external modules) are defined before use. */
class SortModules extends Transform with DependencyAPIMigration {

  override def prerequisites = Seq(Dependency(firrtl.passes.CheckChirrtl))
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.ChirrtlEmitters
  override def invalidates(a: Transform) = false

  override def execute(state: CircuitState): CircuitState = {
    val modulesx = InstanceKeyGraph(state.circuit).moduleOrder.reverse
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }

}
