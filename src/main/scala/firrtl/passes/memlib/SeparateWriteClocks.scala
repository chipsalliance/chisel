// SPDX-License-Identifier: Apache-2.0

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.passes.LowerTypes
import firrtl.options.{Dependency, OptionsException}

/**
  * This transform introduces an intermediate wire on the clock field of each write port of synchronous-read memories
  * that have *multiple* write/readwrite ports and undefined read-under-write collision behavior. Ultimately, the
  * introduction of these intermediate wires does not change which clock net clocks each port; therefore, the purpose of
  * this transform is to help generate Verilog that is more amenable to inference of RAM macros with multiple write
  * ports in FPGA synthesis flows. This change will cause each write and each readwrite port to be emitted in a separate
  * clocked procedure, yielding multiple benefits:
  *
  * 1) Separate write procedures avoid implicitly constraining cross-port read-write and write-write collision behaviors
  * 2) The preference for separate clocked procedures for each write port is explicitly specified by Intel and Xilinx
  *
  * While this feature is not intended to be vendor-specific, inference of *multiple-write* RAM macros from behavioral
  * Verilog or VHDL requires both advanced underlying RAM primitives and advanced synthesis tools. Currently, mapping
  * such memories to programmable devices beyond modern Intel and Xilinx architectures can be prohibitive for users.
  *
  * Though the emission of separate processes for write ports could be absorbed into the Verilog emitter, the use of a
  * pure-FIRRTL transform reduces implementation complexity and enhances reliability.
  */
class SeparateWriteClocks extends Transform with DependencyAPIMigration {
  override def prerequisites = Seq(Dependency(passes.RemoveCHIRRTL), Dependency(passes.ExpandConnects))
  override def optionalPrerequisites = Seq(Dependency[InferReadWrite])
  override def optionalPrerequisiteOf = Seq(Dependency[SetDefaultReadUnderWrite])
  override def invalidates(a: Transform): Boolean = a match {
    case ResolveFlows => true
    case _            => false
  }

  private type ExprMap = collection.mutable.HashMap[WrappedExpression, Reference]

  private def onExpr(replaceExprs: ExprMap)(expr: Expression): Expression = expr match {
    case wsf: WSubField if (replaceExprs.contains(WrappedExpression(wsf))) =>
      replaceExprs(WrappedExpression(wsf))
    case e => e.mapExpr(onExpr(replaceExprs))
  }

  private def isMultiWriteSyncReadUndefinedRUW(mem: DefMemory): Boolean = {
    (mem.writers.size + mem.readwriters.size) > 1 &&
    mem.readLatency == 1 && mem.writeLatency == 1 &&
    mem.readUnderWrite == ReadUnderWrite.Undefined
  }

  private def onStmt(replaceExprs: ExprMap, ns: Namespace)(stmt: Statement): Statement = stmt match {
    case mem: DefMemory if isMultiWriteSyncReadUndefinedRUW(mem) =>
      val clockRefs = (mem.writers ++ mem.readwriters).map { p => MemPortUtils.memPortField(mem, p, "clk") }
      val clockWireMap = clockRefs.map { pClk =>
        WrappedExpression(pClk) -> DefWire(mem.info, ns.newName(LowerTypes.loweredName(pClk)), ClockType)
      }
      val clockStmts = clockWireMap.flatMap {
        case (pClk, clkWire) => Seq(clkWire, Connect(mem.info, pClk.e1, Reference(clkWire)))
      }
      replaceExprs ++= clockWireMap.map { case (pClk, clkWire) => pClk -> Reference(clkWire) }
      Block(mem +: clockStmts)
    case Connect(i, lhs, rhs)        => Connect(i, onExpr(replaceExprs)(lhs), rhs)
    case PartialConnect(i, lhs, rhs) => PartialConnect(i, onExpr(replaceExprs)(lhs), rhs)
    case IsInvalid(i, invalidated)   => IsInvalid(i, onExpr(replaceExprs)(invalidated))
    case s                           => s.mapStmt(onStmt(replaceExprs, ns))
  }

  override def execute(state: CircuitState): CircuitState = {
    val c = state.circuit
    val cPrime = c.copy(modules = c.modules.map(m => m.mapStmt(onStmt(new ExprMap, Namespace(m)))))
    state.copy(circuit = cPrime)
  }
}
