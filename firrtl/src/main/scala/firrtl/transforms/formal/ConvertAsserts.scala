// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms.formal

import firrtl._
import firrtl.ir._
import firrtl.options.Dependency

/** Convert Asserts
  *
  * Replaces all Assert nodes with a gated print-and-stop. This effectively
  * emulates the assert for IEEE 1364 Verilog.
  */
object ConvertAsserts extends Transform with DependencyAPIMigration {
  override def prerequisites = Nil
  override def optionalPrerequisites = Nil
  override def optionalPrerequisiteOf =
    Seq(Dependency[VerilogEmitter], Dependency[MinimumVerilogEmitter], Dependency[RemoveVerificationStatements])

  override def invalidates(a: Transform): Boolean = false

  def convertAsserts(namespace: Namespace, stmt: Statement): Statement = stmt match {
    case v: Verification if v.op == Formal.Assert =>
      val nPred = DoPrim(PrimOps.Not, Seq(v.pred), Nil, v.pred.tpe)
      val gatedNPred = DoPrim(PrimOps.And, Seq(nPred, v.en), Nil, v.pred.tpe)
      val name = if (v.name.nonEmpty) { v.name }
      else { namespace.newName("assert") }
      val stop = Stop(v.info, 1, v.clk, gatedNPred, name)
      v.msg match {
        case StringLit("") => stop
        case msg =>
          val printName = namespace.newName(name + "_print")
          Block(Print(v.info, msg, Nil, v.clk, gatedNPred, printName), stop)
      }
    case s => s.mapStmt(convertAsserts(namespace, _))
  }

  def execute(state: CircuitState): CircuitState = {
    state.copy(circuit = state.circuit.mapModule { m =>
      val namespace = Namespace(m)
      // make sure the name assert is reserved
      if (!namespace.contains("assert")) { namespace.newName("assert") }
      m.mapStmt(convertAsserts(namespace, _))
    })
  }
}
