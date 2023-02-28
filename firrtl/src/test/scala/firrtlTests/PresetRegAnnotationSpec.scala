package firrtlTests

import firrtl._
import firrtl.annotations.{CircuitTarget, PresetRegAnnotation}
import firrtl.options.Dependency
import firrtl.testutils.LeanTransformSpec
import firrtl.transforms.PropagatePresetAnnotations
import logger.{LogLevel, LogLevelAnnotation, Logger}

import scala.collection.mutable

/** Tests the use of the [[firrtl.annotations.PresetRegAnnotation]]
  * from a pass that needs to create a register with an initial value.
  */
class PresetRegAnnotationSpec
    extends LeanTransformSpec(Seq(Dependency(MakePresetRegs), Dependency[SystemVerilogEmitter])) {
  behavior.of("PresetRegAnnotation")

  val src =
    """circuit test:
      |  module test:
      |    input clock : Clock
      |    output out : UInt<8>
      |
      |    reg r : UInt<8>, clock with :
      |      reset => (UInt(0), UInt<8>("h7b"))
      |    out <= r
      |""".stripMargin

  val ll = LogLevel.Error

  it should "allow passes to mark registers to be initialized" in {
    val result = Logger.makeScope(Seq(LogLevelAnnotation(ll))) { compile(src) }
    val verilog = result.getEmittedCircuit.value
    val lines = verilog.split('\n').map(_.trim).toSet

    // the register should be assigned its initial value
    assert(lines.contains("reg [7:0] r = 8'h7b;"))
    assert(lines.contains("assign out = r;"))

    // no asynchronous reset should be emitted
    assert(!lines.contains("if (reset) begin"))
    assert(!lines.contains("always @(posedge clock or posedge reset) begin"))
  }
}

/** Adds preset reg annotations for all registers that have:
  * 1. reset = false
  * 2. init = a literal
  */
private object MakePresetRegs extends Transform with DependencyAPIMigration {
  // run on lowered firrtl
  override def prerequisites = Seq(Dependency(firrtl.passes.ExpandWhens), Dependency(firrtl.passes.LowerTypes))
  override def invalidates(a: Transform) = false
  // since we generate PresetRegAnnotations, we need to run after preset propagation
  override def optionalPrerequisites = Seq(Dependency[PropagatePresetAnnotations])
  // we want to run before the actual Verilog is emitted
  // we want to look at the reset value, which may be removed by the RemoveReset transform.
  override def optionalPrerequisiteOf = Seq(Dependency[SystemVerilogEmitter], Dependency(firrtl.transforms.RemoveReset))

  override def execute(state: CircuitState): CircuitState = {
    val c = CircuitTarget(state.circuit.main)
    val newAnnos = state.circuit.modules.flatMap(onModule(c, _))
    state.copy(annotations = newAnnos ++: state.annotations)
  }

  private def onModule(c: CircuitTarget, m: ir.DefModule): List[PresetRegAnnotation] = m match {
    case mod: ir.Module =>
      val regs = mutable.ListBuffer[String]()
      mod.foreachStmt(onStmt(_, regs))
      val m = c.module(mod.name)
      regs.map(r => PresetRegAnnotation(m.ref(r))).toList
    case _ => List()
  }

  private def onStmt(s: ir.Statement, regs: mutable.ListBuffer[String]): Unit = s match {
    case ir.DefRegister(_, name, _, _, reset, init) if reset == Utils.False() && Utils.isLiteral(init) =>
      regs.append(name)
    case other => other.foreachStmt(onStmt(_, regs))
  }
}
