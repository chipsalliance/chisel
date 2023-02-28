package firrtl

import firrtl.ir.{Expression, Info, StringLit}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class SystemVerilogEmitter extends VerilogEmitter {
  override val outputSuffix: String = ".sv"

  override def prerequisites = firrtl.stage.Forms.LowFormOptimized

  override def addFormalStatement(
    formals: mutable.Map[Expression, ArrayBuffer[Seq[Any]]],
    clk:     Expression,
    en:      Expression,
    stmt:    Seq[Any],
    info:    Info,
    msg:     StringLit
  ): Unit = {
    val lines = formals.getOrElseUpdate(clk, ArrayBuffer[Seq[Any]]())
    lines += Seq("// ", msg.serialize)
    lines += Seq("if (", en, ") begin")
    lines += Seq(tab, stmt, info)
    lines += Seq("end")
  }

  override def execute(state: CircuitState): CircuitState = {
    super.execute(state)
  }
}
