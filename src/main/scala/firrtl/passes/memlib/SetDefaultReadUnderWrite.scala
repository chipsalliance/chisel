// SPDX-License-Identifier: Apache-2.0

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.options.{Dependency, OptionsException}
import firrtl.annotations.NoTargetAnnotation

sealed trait DefaultReadUnderWriteAnnotation extends NoTargetAnnotation

/** This annotation directs the [[SetDefaultReadUnderWrite]] transform to assign a default value of 'old' (read-first
  * behavior) to all synchronous-read memories with 'undefined' read-under-write parameters.
  */
case object DefaultReadFirstAnnotation extends DefaultReadUnderWriteAnnotation

/** This annotation directs the [[SetDefaultReadUnderWrite]] transform to assign a default value of 'new' (write-first
  * behavior) to all synchronous-read memories with 'undefined' read-under-write parameters.
  */
case object DefaultWriteFirstAnnotation extends DefaultReadUnderWriteAnnotation

/**
  * Adding a [[DefaultReadUnderWriteAnnotation]] and running the [[SetDefaultReadUnderWrite]] transform will cause all
  * synchronous-read memories with 'undefined' read-under-write parameters to be assigned a default parameter value,
  * either 'old' (read-first behavior) or 'new' (write-first behavior). This can help generate Verilog that is amenable
  * to RAM macro inference for various FPGA tools, or it can be used to satisfy other downstream design constraints.
  */
class SetDefaultReadUnderWrite extends Transform with DependencyAPIMigration {
  override def prerequisites = firrtl.stage.Forms.HighForm
  override def optionalPrerequisites = Seq(Dependency[InferReadWrite])
  override def optionalPrerequisiteOf = Seq(Dependency(VerilogMemDelays))
  override def invalidates(a: Transform): Boolean = false

  private def onStmt(defaultRUW: ReadUnderWrite.Value)(stmt: Statement): Statement = stmt match {
    case mem: DefMemory if (mem.readLatency > 0 && mem.readUnderWrite == ReadUnderWrite.Undefined) =>
      mem.copy(readUnderWrite = defaultRUW)
    case s => s.mapStmt(onStmt(defaultRUW))
  }

  override def execute(state: CircuitState): CircuitState = {
    val c = state.circuit
    val ruwDefaults = state.annotations
      .collect({
        case DefaultReadFirstAnnotation  => ReadUnderWrite.Old
        case DefaultWriteFirstAnnotation => ReadUnderWrite.New
      })
      .toSet
    if (ruwDefaults.size == 0) {
      state
    } else if (ruwDefaults.size == 1) {
      state.copy(circuit = c.copy(modules = c.modules.map(m => m.mapStmt(onStmt(ruwDefaults.head)))))
    } else {
      throw new OptionsException("Conflicting default read-under-write settings.")
    }
  }
}
