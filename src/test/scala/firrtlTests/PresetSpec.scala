// See LICENSE for license details.

package firrtlTests

import firrtl._
import FirrtlCheckers._
import firrtl.annotations._

class PresetSpec extends FirrtlFlatSpec {
  type Mod = Seq[String]
  type ModuleSeq = Seq[Mod]
  def compile(input: String, annos: AnnotationSeq): CircuitState =
    (new VerilogCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), List.empty)
  def compileBody(modules: ModuleSeq) = {
    val annos = Seq(new PresetAnnotation(CircuitTarget("Test").module("Test").ref("reset")), firrtl.transforms.NoDCEAnnotation)
    var str = """
      |circuit Test :
      |""".stripMargin
    modules foreach ((m: Mod) => {
      val header = "|module " + m(0) + " :"
      str +=  header.stripMargin.stripMargin.split("\n").mkString("  ", "\n  ", "")
      str += m(1).split("\n").mkString("    ", "\n    ", "")
      str += """
        |""".stripMargin
    })
    compile(str,annos)
  }

  "Preset" should """behave properly given a `Preset` annotated `AsyncReset` INPUT reset:
    - replace AsyncReset specific blocks by standard Register blocks 
    - add inline declaration of all registers connected to reset
    - remove the useless input port""" in {
    val result = compileBody(Seq(Seq("Test",s"""
      |input clock : Clock
      |input reset : AsyncReset
      |input x : UInt<1>
      |output z : UInt<1>
      |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
      |r <= x
      |z <= r""".stripMargin))
    )
    result shouldNot containLine ("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines (
      "if (reset) begin",
      "r = 1'h0;",
      "end")
    result shouldNot containLine ("input   reset,")
    result should containLine ("always @(posedge clock) begin")
    result should containLine ("reg  r = 1'h0;")
  }
  
  it should """behave properly given a `Preset` annotated `AsyncReset` WIRE reset:
    - replace AsyncReset specific blocks by standard Register blocks 
    - add inline declaration of all registers connected to reset
    - remove the useless wire declaration and assignation""" in {
    val result = compileBody(Seq(Seq("Test",s"""
      |input clock : Clock
      |input x : UInt<1>
      |output z : UInt<1>
      |wire reset : AsyncReset
      |reset <= asAsyncReset(UInt(0))
      |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
      |r <= x
      |z <= r""".stripMargin))
    )
    result shouldNot containLine ("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines (
      "if (reset) begin",
      "r = 1'h0;",
      "end")
    result should containLine ("always @(posedge clock) begin")
    result should containLine ("reg  r = 1'h0;")
    // it should also remove useless asyncReset signal, all along the path down to registers 
    result shouldNot containLine ("wire  reset;")
    result shouldNot containLine ("assign reset = 1'h0;")
  }
  it should "raise TreeCleanUpOrphantException on cast of annotated AsyncReset" in {
    an [firrtl.transforms.PropagatePresetAnnotations.TreeCleanUpOrphanException] shouldBe thrownBy {
      compileBody(Seq(Seq("Test",s"""
        |input clock : Clock
        |input x : UInt<1>
        |output z : UInt<1>
        |output sz : UInt<1>
        |wire reset : AsyncReset
        |reset <= asAsyncReset(UInt(0))
        |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
        |wire sreset : UInt<1>
        |sreset <= asUInt(reset) ; this is FORBIDDEN
        |reg s : UInt<1>, clock with : (reset => (sreset, UInt(0)))
        |r <= x
        |s <= x
        |z <= r
        |sz <= s""".stripMargin))
      )
    }
  }
  
  it should "propagate through bundles" in {
    val result = compileBody(Seq(Seq("Test",s"""
      |input clock : Clock
      |input reset : AsyncReset
      |input x : UInt<1>
      |output z : UInt<1>
      |wire bundle : {in_rst: AsyncReset, out_rst:AsyncReset}
      |bundle.in_rst <= reset
      |bundle.out_rst <= bundle.in_rst
      |reg r : UInt<1>, clock with : (reset => (bundle.out_rst, UInt(0)))
      |r <= x
      |z <= r""".stripMargin))
    )
    result shouldNot containLine ("input   reset,")
    result shouldNot containLine ("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines (
      "if (reset) begin",
      "r = 1'h0;",
      "end")
    result should containLine ("always @(posedge clock) begin")
    result should containLine ("reg  r = 1'h0;")
  }
  it should "propagate through vectors" in {
    val result = compileBody(Seq(Seq("Test",s"""
      |input clock : Clock
      |input reset : AsyncReset
      |input x : UInt<1>
      |output z : UInt<1>
      |wire vector : AsyncReset[2]
      |vector[0] <= reset
      |vector[1] <= vector[0]
      |reg r : UInt<1>, clock with : (reset => (vector[1], UInt(0)))
      |r <= x
      |z <= r""".stripMargin))
    )
    result shouldNot containLine ("input   reset,")
    result shouldNot containLine ("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines (
      "if (reset) begin",
      "r = 1'h0;",
      "end")
    result should containLine ("always @(posedge clock) begin")
    result should containLine ("reg  r = 1'h0;")
  }
  
  it should "propagate through bundles of vectors" in {
    val result = compileBody(Seq(Seq("Test",s"""
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
      |z <= r""".stripMargin))
    )
    result shouldNot containLine ("input   reset,")
    result shouldNot containLine ("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines (
      "if (reset) begin",
      "r = 1'h0;",
      "end")
    result should containLine ("always @(posedge clock) begin")
    result should containLine ("reg  r = 1'h0;")
  }
  it should """propagate properly accross modules: 
    - replace AsyncReset specific blocks by standard Register blocks 
    - add inline declaration of all registers connected to reset
    - remove the useless input port of instanciated module
    - remove the useless instance connections 
    - remove wires and assignations used in instance connections
    """ in {
    val result = compileBody(Seq(
      Seq("TestA",s"""
      |input clock : Clock
      |input reset : AsyncReset
      |input x : UInt<1>
      |output z : UInt<1>
      |reg r : UInt<1>, clock with : (reset => (reset, UInt(0)))
      |r <= x
      |z <= r
      |""".stripMargin), 
      Seq("Test",s"""
      |input clock : Clock
      |input x : UInt<1>
      |output z : UInt<1>
      |wire reset : AsyncReset
      |reset <= asAsyncReset(UInt(0))
      |inst i of TestA
      |i.clock <= clock
      |i.reset <= reset
      |i.x <= x
      |z <= i.z""".stripMargin)
    ))
    // assess that all useless connections are not emitted
    result shouldNot containLine ("wire i_reset;")
    result shouldNot containLine (".reset(i_reset),")
    result shouldNot containLine ("assign i_reset = reset;")
    result shouldNot containLine ("input   reset,")
    
    result shouldNot containLine ("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines (
      "if (reset) begin",
      "r = 1'h0;",
      "end")
    result should containLine ("always @(posedge clock) begin")
    result should containLine ("reg  r = 1'h0;")
  }

  it should "propagate even through disordonned statements" in {
    val result = compileBody(Seq(Seq("Test",s"""
      |input clock : Clock
      |input reset : AsyncReset
      |input x : UInt<1>
      |output z : UInt<1>
      |wire bundle : {in_rst: AsyncReset, out_rst:AsyncReset}
      |reg r : UInt<1>, clock with : (reset => (bundle.out_rst, UInt(0)))
      |bundle.out_rst <= bundle.in_rst
      |bundle.in_rst <= reset
      |r <= x
      |z <= r""".stripMargin))
    )
    result shouldNot containLine ("input   reset,")
    result shouldNot containLine ("always @(posedge clock or posedge reset) begin")
    result shouldNot containLines (
      "if (reset) begin",
      "r = 1'h0;",
      "end")
    result should containLine ("always @(posedge clock) begin")
    result should containLine ("reg  r = 1'h0;")
  }

}

class PresetExecutionTest extends ExecutionTest(
  "PresetTester", 
  "/features", 
  annotations = Seq(new PresetAnnotation(CircuitTarget("PresetTester").module("PresetTester").ref("preset")))
)
