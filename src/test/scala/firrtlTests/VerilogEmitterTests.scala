// See LICENSE for license details.

package firrtlTests

import java.io._

import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.annotations._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.transforms.VerilogRename
import firrtl.Parser.IgnoreInfo
import FirrtlCheckers._
import firrtl.transforms.CombineCats

class DoPrimVerilog extends FirrtlFlatSpec {
  "Xorr" should "emit correctly" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Xorr : 
        |  module Xorr : 
        |    input a: UInt<4>
        |    output b: UInt<1>
        |    b <= xorr(a)""".stripMargin
    val check = 
      """module Xorr(
        |  input  [3:0] a,
        |  output  b
        |);
        |  assign b = ^a;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }
  "Andr" should "emit correctly" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Andr : 
        |  module Andr : 
        |    input a: UInt<4>
        |    output b: UInt<1>
        |    b <= andr(a)""".stripMargin
    val check = 
      """module Andr(
        |  input  [3:0] a,
        |  output  b
        |);
        |  assign b = &a;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }
  "Orr" should "emit correctly" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Orr : 
        |  module Orr : 
        |    input a: UInt<4>
        |    output b: UInt<1>
        |    b <= orr(a)""".stripMargin
    val check = 
      """module Orr(
        |  input  [3:0] a,
        |  output  b
        |);
        |  assign b = |a;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }
  "Rem" should "emit correctly" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input in : UInt<8>
        |    output out : UInt<1>
        |    out <= rem(in, UInt<1>("h1"))
        |""".stripMargin
    val check =
      """module Test(
        |  input  [7:0] in, 
        |  output  out 
        |);
        |  wire [7:0] _GEN_0;
        |  assign out = _GEN_0[0];
        |  assign _GEN_0 = in % 8'h1;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }
  "nested cats" should "emit correctly" in {
    val compiler = new MinimumVerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input in1 : UInt<1>
        |    input in2 : UInt<2>
        |    input in3 : UInt<3>
        |    input in4 : UInt<4>
        |    output out : UInt<10>
        |    out <= cat(in4, cat(in3, cat(in2, in1)))
        |""".stripMargin
    val check =
      """module Test(
        |  input  in1,
        |  input  [1:0] in2,
        |  input  [2:0] in3,
        |  input  [3:0] in4,
        |  output [9:0] out
        |);
        |  wire [5:0] _GEN_1;
        |  assign out = {in4,_GEN_1};
        |  assign _GEN_1 = {in3,in2,in1};
        |endmodule
        |""".stripMargin.split("\n") map normalized

    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm), Seq(new CombineCats()))
    val lines = finalState.getEmittedCircuit.value split "\n" map normalized
    for (e <- check) {
      lines should contain (e)
    }
  }
}

class VerilogEmitterSpec extends FirrtlFlatSpec {
  "Ports" should "emit with widths aligned and names aligned" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input a : UInt<25000>
        |    output b : UInt
        |    input c : UInt<32>
        |    output d : UInt
        |    input e : UInt<1>
        |    input f : Analog<32>
        |    b <= a
        |    d <= add(c, e)
        |""".stripMargin
    val check = Seq(
      "  input  [24999:0] a,",
      "  output [24999:0] b,",
      "  input  [31:0]    c,",
      "  output [32:0]    d,",
      "  input            e,",
      "  inout  [31:0]    f"
    )
    // We don't use executeTest because we care about the spacing in the result
    val writer = new java.io.StringWriter
    compiler.compile(CircuitState(parse(input), ChirrtlForm), writer)
    val lines = writer.toString.split("\n")
    for (c <- check) {
      lines should contain (c)
    }
  }
  "The Verilog Emitter" should "support Modules with no ports" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    wire x : UInt<32>
        |    x <= UInt(0)
      """.stripMargin
    compiler.compile(CircuitState(parse(input), ChirrtlForm), new java.io.StringWriter)
  }
  "AsClock" should "emit correctly" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input in : UInt<1>
        |    output out : Clock
        |    out <= asClock(in)
        |""".stripMargin
    val check =
      """module Test(
        |  input   in,
        |  output  out
        |);
        |  assign out = in;
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, check, compiler)
  }
  "The Verilog Emitter" should "support pads with width <= the width of the argument" in {
    // We do just a few passes instead of using the VerilogCompiler to ensure that the pad actually
    // reaches the VerilogEmitter and isn't removed by an optimization transform
    val passes = Seq(
      ToWorkingIR,
      ResolveKinds,
      InferTypes
    )
    def input(n: Int) =
      s"""circuit Test :
         |  module Test :
         |    input in : UInt<8>
         |    output out : UInt<8>
         |    out <= pad(in, $n)
         |""".stripMargin
    for (w <- Seq(6, 8)) {
      val circuit = passes.foldLeft(parse(input(w))) { case (c, p) => p.run(c) }
      val state = CircuitState(circuit, LowForm, Seq(EmitCircuitAnnotation(classOf[VerilogEmitter])))
      val emitter = new VerilogEmitter
      val result = emitter.execute(state)
      result should containLine ("assign out = in;")
    }
  }

  "The verilog emitter" should "offer support for generating bindable forms of modules" in {
    val emitter = new VerilogEmitter
    val input =
      """circuit Test :
        |  module Test :
        |    input a : UInt<25000>
        |    output b : UInt
        |    input c : UInt<32>
        |    output d : UInt
        |    input e : UInt<1>
        |    input f : Analog<32>
        |    b <= a
        |    d <= add(c, e)
        |""".stripMargin
    val check =
      """
        |module BindsToTest(
        |  input  [24999:0] a,
        |  output [24999:0] b,
        |  input  [31:0]    c,
        |  output [32:0]    d,
        |  input            e,
        |  inout  [31:0]    f
        |);
        |
        |$readmemh("file", memory);
        |
        |endmodule""".stripMargin.split("\n")

    // We don't use executeTest because we care about the spacing in the result
    val writer = new java.io.StringWriter

    val initialState = CircuitState(parse(input), ChirrtlForm)
    val compiler = new LowFirrtlCompiler()

    val state = compiler.compile(initialState, Seq.empty)

    val moduleMap = state.circuit.modules.map(m => m.name -> m).toMap

    val module = state.circuit.modules.filter(module => module.name == "Test").collectFirst { case m: firrtl.ir.Module => m }.get

    val renderer = emitter.getRenderer(module, moduleMap)(writer)

    renderer.emitVerilogBind("BindsToTest",
      """
        |$readmemh("file", memory);
        |
        |""".stripMargin)
    val lines = writer.toString.split("\n")

    val outString = writer.toString

    // This confirms that the module io's were emitted
    for (c <- check) {
      lines should contain (c)
    }
  }

  "Verilog name conflicts" should "be resolved" in {
    val input =
      """|circuit parameter:
         |  module parameter:
         |    input always: UInt<1>
         |    output always$: UInt<1>
         |    inst assign of endmodule
         |    node always_ = not(always)
         |    node always__ = and(always_, assign.fork)
         |    always$ <= always__
         |  module endmodule:
         |    output fork: UInt<1>
         |    node const = add(UInt<4>("h1"), UInt<3>("h2"))
         |    fork <= const
         |""".stripMargin
    val check_firrtl =
      """|circuit parameter_:
         |  module parameter_:
         |    input always___: UInt<1>
         |    output always$: UInt<1>
         |    inst assign_ of endmodule_
         |    node always_ = not(always___)
         |    node always__ = and(always_, assign_.fork_)
         |    always$ <= always__
         |  module endmodule_:
         |    output fork_: UInt<1>
         |    node const_ = add(UInt<4>("h1"), UInt<3>("h2"))
         |    fork_ <= const_
         |""".stripMargin
    val state = CircuitState(parse(input), UnknownForm, Seq.empty, None)
    val output = Seq( ToWorkingIR, ResolveKinds, InferTypes, new VerilogRename )
      .foldLeft(state){ case (c, tx) => tx.runTransform(c) }
    Seq( CheckHighForm )
      .foldLeft(output.circuit){ case (c, tx) => tx.run(c) }
    output.circuit.serialize should be (parse(check_firrtl).serialize)
  }

}

class VerilogDescriptionEmitterSpec extends FirrtlFlatSpec {
  "Port descriptions" should "emit aligned comments on the line above" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input a : UInt<1>
        |    input b : UInt<1>
        |    output c : UInt<1>
        |    c <= add(a, b)
        |""".stripMargin
    val check = Seq(
      """  /* multi
        |   * line
        |   */
        |  input   a,""".stripMargin,
      """  // single line
        |  input   b,""".stripMargin
    )
    // We don't use executeTest because we care about the spacing in the result
    val modName = ModuleName("Test", CircuitName("Test"))
    val annos = Seq(
      DescriptionAnnotation(ComponentName("a", modName), "multi\nline"),
      DescriptionAnnotation(ComponentName("b", modName), "single line"))
    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
    val output = finalState.getEmittedCircuit.value
    for (c <- check) {
      assert(output.contains(c))
    }
  }

  "Declaration descriptions" should "emit aligned comments on the line above" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input a : UInt<1>
        |    input b : UInt<1>
        |    output c : UInt<1>
        |
        |    wire d : UInt<1>
        |    d <= add(a, b)
        |
        |    reg e : UInt<1>, clock
        |    e <= or(a, b)
        |
        |    node f = and(a, b)
        |    c <= add(d, add(e, f))
        |""".stripMargin
    val check = Seq(
      """  /* multi
        |   * line
        |   */
        |  wire  d;""".stripMargin,
      """  /* multi
        |   * line
        |   */
        |  reg  e;""".stripMargin,
      """  // single line
        |  wire  f;""".stripMargin
    )
    // We don't use executeTest because we care about the spacing in the result
    val modName = ModuleName("Test", CircuitName("Test"))
    val annos = Seq(
      DescriptionAnnotation(ComponentName("d", modName), "multi\nline"),
      DescriptionAnnotation(ComponentName("e", modName), "multi\nline"),
      DescriptionAnnotation(ComponentName("f", modName), "single line"))
    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
    val output = finalState.getEmittedCircuit.value
    for (c <- check) {
      assert(output.contains(c))
    }
  }

  "Module descriptions" should "emit aligned comments on the line above" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input a : UInt<1>
        |    input b : UInt<1>
        |    output c : UInt<1>
        |
        |    wire d : UInt<1>
        |    d <= add(a, b)
        |
        |    reg e : UInt<1>, clock
        |    e <= or(a, b)
        |
        |    node f = and(a, b)
        |    c <= add(d, add(e, f))
        |""".stripMargin
    val check = Seq(
      """/* multi
        | * line
        | */
        |module Test(""".stripMargin
    )
    // We don't use executeTest because we care about the spacing in the result
    val modName = ModuleName("Test", CircuitName("Test"))
    val annos = Seq(DescriptionAnnotation(modName, "multi\nline"))
    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
    val output = finalState.getEmittedCircuit.value
    for (c <- check) {
      assert(output.contains(c))
    }
  }

  "Multiple descriptions" should "be combined" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Test :
        |  module Test :
        |    input a : UInt<1>
        |    input b : UInt<1>
        |    output c : UInt<1>
        |
        |    wire d : UInt<1>
        |    d <= add(a, b)
        |
        |    c <= add(a, d)
        |""".stripMargin
    val check = Seq(
      """/* line1
        | *
        | * line2
        | */
        |module Test(""".stripMargin,
      """  /* line3
        |   *
        |   * line4
        |   */
        |  input   a,""".stripMargin,
      """  /* line5
        |   *
        |   * line6
        |   */
        |  wire  d;""".stripMargin
    )
    // We don't use executeTest because we care about the spacing in the result
    val modName = ModuleName("Test", CircuitName("Test"))
    val annos = Seq(
      DescriptionAnnotation(modName, "line1"),
      DescriptionAnnotation(modName, "line2"),
      DescriptionAnnotation(ComponentName("a", modName), "line3"),
      DescriptionAnnotation(ComponentName("a", modName), "line4"),
      DescriptionAnnotation(ComponentName("d", modName), "line5"),
      DescriptionAnnotation(ComponentName("d", modName), "line6")
    )
    val writer = new java.io.StringWriter
    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
    val output = finalState.getEmittedCircuit.value
    for (c <- check) {
      assert(output.contains(c))
    }
  }
}
