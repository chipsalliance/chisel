// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.annotations._
import firrtl.testutils._
import firrtl.testutils.FirrtlCheckers._
import logger.{LogLevel, LogLevelAnnotation, Logger}

class PresetSpec extends VerilogTransformSpec {
  type Mod = Seq[String]
  type ModuleSeq = Seq[Mod]

  def compileBody(modules: ModuleSeq) = {
    val annos =
      Seq(PresetAnnotation(CircuitTarget("Test").module("Test").ref("reset")), firrtl.transforms.NoDCEAnnotation)
    var str = """
                |circuit Test :
                |""".stripMargin
    modules.foreach((m: Mod) => {
      val header = "|module " + m(0) + " :"
      str += header.stripMargin.stripMargin.split("\n").mkString("  ", "\n  ", "")
      str += m(1).split("\n").mkString("    ", "\n    ", "")
      str += """
               |""".stripMargin
    })
    val logLevel = LogLevel.Warn
    Logger.makeScope(Seq(LogLevelAnnotation(logLevel))) {
      compile(str, annos)
    }
  }

  "Preset" should """behave properly given a `Preset` annotated `AsyncReset` INPUT reset:
    - replace AsyncReset specific blocks by standard Register blocks 
    - add inline declaration of all registers connected to reset
    - remove the useless input port""" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input reset : AsyncReset
             |input x : UInt<1>
             |output z : UInt<1>
             |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
             |r <= x
             |z <= r""".stripMargin
        )
      )
    )
    result shouldNot containLine("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines("if (reset) begin", "r = 1'h0;", "end")
    result shouldNot containLine("input   reset,")
    result should containLine("always @(posedge clock) begin")
    result should containLine("reg  r = 1'h0;")
  }

  it should """behave properly given a `Preset` annotated `AsyncReset` WIRE reset:
    - replace AsyncReset specific blocks by standard Register blocks 
    - add inline declaration of all registers connected to reset
    - remove the useless wire declaration and assignation""" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input x : UInt<1>
             |output z : UInt<1>
             |wire reset : AsyncReset
             |reset <= asAsyncReset(UInt(0))
             |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
             |r <= x
             |z <= r""".stripMargin
        )
      )
    )

    result shouldNot containLine("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines("if (reset) begin", "r = 1'h0;", "end")
    result should containLine("always @(posedge clock) begin")
    result should containLine("reg  r = 1'h0;")
    // it should also remove useless asyncReset signal, all along the path down to registers
    result shouldNot containLine("wire  reset;")
    result shouldNot containLine("assign reset = 1'h0;")
  }
  it should "replace usages of the preset reset with the constant 0 since the reset is never active" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input x : UInt<1>
             |output z : UInt<1>
             |output sz : UInt<1>
             |wire reset : AsyncReset
             |reset <= asAsyncReset(UInt(0))
             |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
             |wire sreset : UInt<1>
             |sreset <= asUInt(reset) ; this is ok, essentially like assigning zero to the wire
             |reg s : UInt<1>, clock with : (reset => (sreset, UInt(0)))
             |r <= x
             |s <= x
             |z <= r
             |sz <= s""".stripMargin
        )
      )
    )

    result should containLine("wire  sreset = 1'h0;")
  }

  it should "propagate through bundles" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input reset : AsyncReset
             |input x : UInt<1>
             |output z : UInt<1>
             |wire bundle : {in_rst: AsyncReset, out_rst:AsyncReset}
             |bundle.in_rst <= reset
             |bundle.out_rst <= bundle.in_rst
             |reg r : UInt<1>, clock with : (reset => (bundle.out_rst, UInt(0)))
             |r <= x
             |z <= r""".stripMargin
        )
      )
    )
    result shouldNot containLine("input   reset,")
    result shouldNot containLine("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines("if (reset) begin", "r = 1'h0;", "end")
    result should containLine("always @(posedge clock) begin")
    result should containLine("reg  r = 1'h0;")
  }
  it should "propagate through vectors" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input reset : AsyncReset
             |input x : UInt<1>
             |output z : UInt<1>
             |wire vector : AsyncReset[2]
             |vector[0] <= reset
             |vector[1] <= vector[0]
             |reg r : UInt<1>, clock with : (reset => (vector[1], UInt(0)))
             |r <= x
             |z <= r""".stripMargin
        )
      )
    )
    result shouldNot containLine("input   reset,")
    result shouldNot containLine("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines("if (reset) begin", "r = 1'h0;", "end")
    result should containLine("always @(posedge clock) begin")
    result should containLine("reg  r = 1'h0;")
  }

  it should "propagate through bundles of vectors" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input reset : AsyncReset
             |input x : UInt<1>
             |output z : UInt<1>
             |wire bundle : {in_rst: AsyncReset[2], out_rst:AsyncReset}
             |bundle.in_rst[0] <= reset
             |bundle.in_rst[1] <= bundle.in_rst[0]
             |bundle.out_rst <= bundle.in_rst[1]
             |reg r : UInt<1>, clock with : (reset => (bundle.out_rst, UInt(0)))
             |r <= x
             |z <= r""".stripMargin
        )
      )
    )
    result shouldNot containLine("input   reset,")
    result shouldNot containLine("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines("if (reset) begin", "r = 1'h0;", "end")
    result should containLine("always @(posedge clock) begin")
    result should containLine("reg  r = 1'h0;")
  }
  it should """propagate properly accross modules: 
    - replace AsyncReset specific blocks by standard Register blocks 
    - add inline declaration of all registers connected to reset
    - remove the useless input port of instanciated module
    - remove the useless instance connections 
    - remove wires and assignations used in instance connections
    """ in {
    val result = compileBody(
      Seq(
        Seq(
          "TestA",
          s"""
             |input clock : Clock
             |input reset : AsyncReset
             |input x : UInt<1>
             |output z : UInt<1>
             |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
             |r <= x
             |z <= r
             |""".stripMargin
        ),
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input x : UInt<1>
             |output z : UInt<1>
             |wire reset : AsyncReset
             |reset <= asAsyncReset(UInt(0))
             |inst i of TestA
             |i.clock <= clock
             |i.reset <= reset
             |i.x <= x
             |z <= i.z""".stripMargin
        )
      )
    )
    // assess that all useless connections are not emitted
    result shouldNot containLine("wire i_reset;")
    result shouldNot containLine(".reset(i_reset),")
    result shouldNot containLine("assign i_reset = reset;")
    result shouldNot containLine("input   reset,")

    result shouldNot containLine("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines("if (reset) begin", "r = 1'h0;", "end")
    result should containLine("always @(posedge clock) begin")
    result should containLine("reg  r = 1'h0;")
  }

  it should "propagate zeros for all other uses of an async reset" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input reset : AsyncReset
             |output a : UInt<2>
             |output b : UInt<2>
             |
             |node t = reset
             |node not_t = not(asUInt(reset))
             |a <= cat(asUInt(t), not_t)
             |node t2 = asUInt(t)
             |node t3 = asAsyncReset(t2)
             |b <= cat(asUInt(asSInt(reset)), asUInt(t3))
             |""".stripMargin
        )
      )
    )

    result should containLine("assign b = {1'h0,1'h0};")
  }

  it should "propagate even through disordonned statements" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input reset : AsyncReset
             |input x : UInt<1>
             |output z : UInt<1>
             |wire bundle : {in_rst: AsyncReset, out_rst:AsyncReset}
             |reg r : UInt<1>, clock with : (reset => (bundle.out_rst, UInt(0)))
             |bundle.out_rst <= bundle.in_rst
             |bundle.in_rst <= reset
             |r <= x
             |z <= r""".stripMargin
        )
      )
    )
    result shouldNot containLine("input   reset,")
    result shouldNot containLine("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines("if (reset) begin", "r = 1'h0;", "end")
    result should containLine("always @(posedge clock) begin")
    result should containLine("reg  r = 1'h0;")
  }

  it should "work with verification statements that are guarded by a preset reset" in {
    val result = compileBody(
      Seq(
        Seq(
          "Test",
          s"""
             |input clock : Clock
             |input in : UInt<4>
             |input reset : AsyncReset
             |
             |node _T = eq(in, UInt<2>("h3")) @[main.scala 19:15]
             |node _T_1 = asUInt(reset) @[main.scala 19:11]
             |node _T_2 = eq(_T_1, UInt<1>("h0")) @[main.scala 19:11]
             |when _T_2 : @[main.scala 19:11]
             |  assert(clock, _T, UInt<1>("h1"), "") : assert @[main.scala 19:11]
             |  node _T_3 = eq(_T, UInt<1>("h0")) @[main.scala 19:11]
             |  when _T_3 : @[main.scala 19:11]
             |    printf(clock, UInt<1>("h1"), "Assertion failed") : printf @[main.scala 19:11]
             |
             |""".stripMargin
        )
      )
    )
    // just getting here without falling over the fact that `reset` gets removed is great!
  }

}

class PresetExecutionTest
    extends ExecutionTest(
      "PresetTester",
      "/features",
      annotations = Seq(new PresetAnnotation(CircuitTarget("PresetTester").module("PresetTester").ref("preset")))
    )
