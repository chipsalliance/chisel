// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import java.io.File

import firrtl._
import firrtl.stage._
import firrtl.annotations._
import firrtl.passes._
import firrtl.transforms.{CombineCats, NoDCEAnnotation}
import firrtl.testutils._
import firrtl.testutils.FirrtlCheckers._
import firrtl.util.BackendCompilationUtilities

import scala.sys.process.{Process, ProcessLogger}

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
        |""".stripMargin.split("\n").map(normalized)
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
        |""".stripMargin.split("\n").map(normalized)
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
        |""".stripMargin.split("\n").map(normalized)
    executeTest(input, check, compiler)
  }
  "Not" should "emit correctly" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit Not :
        |  module Not :
        |    input a: UInt<1>
        |    output b: UInt<1>
        |    b <= not(a)""".stripMargin
    val check =
      """module Not(
        |  input   a,
        |  output  b
        |);
        |  assign b = ~a;
        |endmodule
        |""".stripMargin.split("\n").map(normalized)
    executeTest(input, check, compiler)
  }
  "inline Bits" should "emit correctly" in {
    val compiler = new VerilogCompiler
    val input =
      """circuit InlineBits :
        |  module InlineBits :
        |    input a: UInt<4>
        |    output b: UInt<1>
        |    output c: UInt<3>
        |    output d: UInt<2>
        |    output e: UInt<2>
        |    output f: UInt<2>
        |    output g: UInt<2>
        |    output h: UInt<2>
        |    output i: UInt<2>
        |    output j: UInt<2>
        |    output k: UInt<1>
        |    output l: UInt<1>
        |    output m: UInt<1>
        |    output n: UInt<1>
        |    output o: UInt<2>
        |    output p: UInt<2>
        |    output q: UInt<2>
        |    output r: UInt<1>
        |    output s: UInt<2>
        |    output t: UInt<2>
        |    output u: UInt<1>
        |    b <= bits(a, 2, 2)
        |    c <= bits(a, 3, 1)
        |    d <= head(a, 2)
        |    e <= tail(a, 2)
        |    f <= bits(bits(a, 3, 1), 2, 1)
        |    g <= bits(head(a, 3), 1, 0)
        |    h <= bits(tail(a, 1), 1, 0)
        |    i <= bits(shr(a, 1), 1, 0)
        |    j <= head(bits(a, 3, 1), 2)
        |    k <= head(head(a, 3), 1)
        |    l <= head(tail(a, 1), 1)
        |    m <= head(shr(a, 1), 1)
        |    n <= tail(bits(a, 3, 1), 2)
        |    o <= tail(head(a, 3), 1)
        |    p <= tail(tail(a, 1), 1)
        |    q <= tail(shr(a, 1), 1)
        |    r <= shr(bits(a, 1, 0), 1)
        |    s <= shr(head(a, 3), 1)
        |    t <= shr(tail(a, 1), 1)
        |    u <= shr(shr(a, 1), 2)""".stripMargin
    val check =
      """module InlineBits(
        |  input  [3:0] a,
        |  output  b,
        |  output [2:0] c,
        |  output [1:0] d,
        |  output [1:0] e,
        |  output [1:0] f,
        |  output [1:0] g,
        |  output [1:0] h,
        |  output [1:0] i,
        |  output [1:0] j,
        |  output  k,
        |  output  l,
        |  output  m,
        |  output  n,
        |  output [1:0] o,
        |  output [1:0] p,
        |  output [1:0] q,
        |  output  r,
        |  output [1:0] s,
        |  output [1:0] t,
        |  output  u
        |);
        |  assign b = a[2];
        |  assign c = a[3:1];
        |  assign d = a[3:2];
        |  assign e = a[1:0];
        |  assign f = a[3:2];
        |  assign g = a[2:1];
        |  assign h = a[1:0];
        |  assign i = a[2:1];
        |  assign j = a[3:2];
        |  assign k = a[3];
        |  assign l = a[2];
        |  assign m = a[3];
        |  assign n = a[1];
        |  assign o = a[2:1];
        |  assign p = a[1:0];
        |  assign q = a[2:1];
        |  assign r = a[1];
        |  assign s = a[3:2];
        |  assign t = a[2:1];
        |  assign u = a[3];
        |endmodule
        |""".stripMargin.split("\n").map(normalized)
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
        |  wire [7:0] _GEN_0 = in % 8'h1;
        |  assign out = _GEN_0[0];
        |endmodule
        |""".stripMargin.split("\n").map(normalized)
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
        |  wire [5:0] _GEN_1 = {in3,in2,in1};
        |  assign out = {in4,_GEN_1};
        |endmodule
        |""".stripMargin.split("\n").map(normalized)

    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm), Seq(new CombineCats()))
    val lines = finalState.getEmittedCircuit.value.split("\n").map(normalized)
    for (e <- check) {
      lines should contain(e)
    }
  }
}

class VerilogEmitterSpec extends FirrtlFlatSpec {
  private def compile(input: String): CircuitState =
    (new VerilogCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), List.empty)
  private def compileBody(body: String): CircuitState = {
    val str = """
                |circuit Test :
                |  module Test :
                |""".stripMargin + body.split("\n").mkString("    ", "\n    ", "")
    compile(str)
  }

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
      lines should contain(c)
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
        |""".stripMargin.split("\n").map(normalized)
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
      result should containLine("assign out = in;")
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

    val module =
      state.circuit.modules.filter(module => module.name == "Test").collectFirst { case m: firrtl.ir.Module => m }.get

    val renderer = emitter.getRenderer(module, moduleMap)(writer)

    renderer.emitVerilogBind(
      "BindsToTest",
      """
        |$readmemh("file", memory);
        |
        |""".stripMargin
    )
    val lines = writer.toString.split("\n")

    val outString = writer.toString

    // This confirms that the module io's were emitted
    for (c <- check) {
      lines should contain(c)
    }
  }

  "Initial Blocks" should "be guarded by ifndef SYNTHESIS and user-defined optional macros" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input reset : AsyncReset
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    reg r : UInt<8>, clock with : (reset => (reset, UInt(0)))
        |    r <= in
        |    out <= r
        """.stripMargin
    val state = CircuitState(parse(input), ChirrtlForm)
    val result = (new VerilogCompiler).compileAndEmit(state, List())
    result should containLines(
      "`ifndef SYNTHESIS",
      "`ifdef FIRRTL_BEFORE_INITIAL",
      "`FIRRTL_BEFORE_INITIAL",
      "`endif",
      "initial begin"
    )
    result should containLines(
      "end // initial",
      "`ifdef FIRRTL_AFTER_INITIAL",
      "`FIRRTL_AFTER_INITIAL",
      "`endif",
      "`endif // SYNTHESIS"
    )
  }

  "Verilog name conflicts" should "be resolved" in {
    val input =
      """|circuit parameter:
         |  module parameter:
         |    input always: UInt<1>
         |    output always$: UInt<1>
         |    inst assign of endmodule
         |    inst edge of endmodule_
         |    node always_ = not(always)
         |    node always__ = and(always_, assign.fork)
         |    node always___ = and(always__, edge.fork)
         |    always$ <= always___
         |  module endmodule:
         |    output fork: UInt<1>
         |    node const = add(UInt<4>("h1"), UInt<3>("h2"))
         |    fork <= const
         |  module endmodule_:
         |    output fork: UInt<1>
         |    node const = add(UInt<4>("h1"), UInt<3>("h1"))
         |    fork <= const
         |""".stripMargin
    val check =
      """module parameter_(
        |  input   always____,
        |  output  always$
        |);
        |  wire  assign__fork_;
        |  wire  edge__fork_;
        |  wire  always_ = ~always____;
        |  wire  always__ = always_;
        |  wire  always___ = 1'h0;
        |  endmodule__ assign_ (
        |    .fork_(assign__fork_)
        |  );
        |  endmodule_ edge_ (
        |    .fork_(edge__fork_)
        |  );
        |  assign always$ = 1'h0;
        |endmodule
        |module endmodule__(
        |  output  fork_
        |);
        |  wire [4:0] const_ = 5'h3;
        |  assign fork_ = 1'h1;
        |endmodule
        |module endmodule_(
        |  output  fork_
        |);
        |  wire [4:0] const_ = 5'h2;
        |  assign fork_ = 1'h0;
        |endmodule
        |""".stripMargin.split('\n')
    val circuit = parse(input)
    val annotations = (new FirrtlStage).transform(
      Seq(
        FirrtlCircuitAnnotation(circuit),
        NoDCEAnnotation,
        EmitCircuitAnnotation(classOf[VerilogEmitter])
      )
    )
    CircuitState(
      annotations.collectFirst {
        case FirrtlCircuitAnnotation(circuit) => circuit
      }.get,
      annotations
    ) should containLines(check: _*)
  }

  behavior.of("Register Updates")

  they should "emit using 'else if' constructs" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    input clock: Clock
          |    input sel: UInt<2>
          |    input in_0: UInt<1>
          |    input in_1: UInt<1>
          |    input in_2: UInt<1>
          |    input in_3: UInt<1>
          |    output out: UInt<1>
          |    reg tmp: UInt<1>, clock
          |    node _GEN_0 = mux(eq(sel, UInt<2>(2)), in_2, in_3)
          |    node _GEN_1 = mux(eq(sel, UInt<2>(1)), in_1, _GEN_0)
          |    tmp <= mux(eq(sel, UInt<2>(0)), in_0, _GEN_1)
          |    out <= tmp
          |""".stripMargin
    val circuit = Seq(ToWorkingIR, ResolveKinds, InferTypes).foldLeft(parse(input)) { case (c, p) => p.run(c) }
    val state = CircuitState(circuit, LowForm, Seq(EmitCircuitAnnotation(classOf[VerilogEmitter])))
    val result = (new VerilogEmitter).execute(state)
    result should containLine("if (sel == 2'h0) begin")
    result should containLine("end else if (sel == 2'h1) begin")
    result should containLine("end else if (sel == 2'h2) begin")
    result should containLine("end else begin")
  }

  they should "ignore self assignments in false conditions" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    input clock: Clock
          |    input sel: UInt<1>
          |    input in: UInt<1>
          |    output out: UInt<1>
          |    reg tmp: UInt<1>, clock
          |    tmp <= mux(eq(sel, UInt<1>(0)), in, tmp)
          |    out <= tmp
          |""".stripMargin
    val circuit = Seq(ToWorkingIR, ResolveKinds, InferTypes).foldLeft(parse(input)) { case (c, p) => p.run(c) }
    val state = CircuitState(circuit, LowForm, Seq(EmitCircuitAnnotation(classOf[VerilogEmitter])))
    val result = (new VerilogEmitter).execute(state)
    result should not(containLine("tmp <= tmp"))
  }

  they should "ignore self assignments in true conditions and invert condition" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    input clock: Clock
          |    input sel: UInt<1>
          |    input in: UInt<1>
          |    output out: UInt<1>
          |    reg tmp: UInt<1>, clock
          |    tmp <= mux(eq(sel, UInt<1>(0)), tmp, in)
          |    out <= tmp
          |""".stripMargin
    val circuit = Seq(ToWorkingIR, ResolveKinds, InferTypes).foldLeft(parse(input)) { case (c, p) => p.run(c) }
    val state = CircuitState(circuit, LowForm, Seq(EmitCircuitAnnotation(classOf[VerilogEmitter])))
    val result = (new VerilogEmitter).execute(state)
    result should containLine("if (!(sel == 1'h0)) begin")
    result should not(containLine("tmp <= tmp"))
  }

  they should "ignore self assignments in both true and false conditions" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    input clock: Clock
          |    input sel: UInt<1>
          |    input in: UInt<1>
          |    output out: UInt<1>
          |    reg tmp: UInt<1>, clock
          |    tmp <= mux(eq(sel, UInt<1>(0)), tmp, tmp)
          |    out <= tmp
          |""".stripMargin
    val circuit = Seq(ToWorkingIR, ResolveKinds, InferTypes).foldLeft(parse(input)) { case (c, p) => p.run(c) }
    val state = CircuitState(circuit, LowForm, Seq(EmitCircuitAnnotation(classOf[VerilogEmitter])))
    val result = (new VerilogEmitter).execute(state)
    result should not(containLine("tmp <= tmp"))
    result should not(containLine("always @(posedge clock) begin"))
  }

  they should "properly indent muxes in either the true or false condition" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    input clock: Clock
          |    input reset: UInt<1>
          |    input sel: UInt<3>
          |    input in_0: UInt<1>
          |    input in_1: UInt<1>
          |    input in_2: UInt<1>
          |    input in_3: UInt<1>
          |    input in_4: UInt<1>
          |    input in_5: UInt<1>
          |    input in_6: UInt<1>
          |    input in_7: UInt<1>
          |    input in_8: UInt<1>
          |    input in_9: UInt<1>
          |    input in_10: UInt<1>
          |    input in_11: UInt<1>
          |    output out: UInt<1>
          |    reg tmp: UInt<0>, clock
          |    node m7 = mux(eq(sel, UInt<3>(7)), in_8, in_9)
          |    node m6 = mux(eq(sel, UInt<3>(6)), in_6, in_7)
          |    node m5 = mux(eq(sel, UInt<3>(5)), in_4, in_5)
          |    node m4 = mux(eq(sel, UInt<3>(4)), in_2, in_3)
          |    node m3 = mux(eq(sel, UInt<3>(3)), in_0, in_1)
          |    node m2 = mux(eq(sel, UInt<3>(2)), m6, m7)
          |    node m1 = mux(eq(sel, UInt<3>(1)), m4, m5)
          |    node m0 = mux(eq(sel, UInt<3>(0)), m2, m3)
          |    tmp <= mux(reset, m0, m1)
          |    out <= tmp
          |""".stripMargin
    val circuit = Seq(ToWorkingIR, ResolveKinds, InferTypes).foldLeft(parse(input)) { case (c, p) => p.run(c) }
    val state = CircuitState(circuit, LowForm, Seq(EmitCircuitAnnotation(classOf[VerilogEmitter])))
    val result = (new VerilogEmitter).execute(state)
    /* The Verilog string is used to check for no whitespace between "else" and "if". */
    val verilogString = result.getEmittedCircuit.value
    result should containLine("if (sel == 3'h0) begin")
    verilogString should include("end else if (sel == 3'h1) begin")
    result should containLine("if (sel == 3'h2) begin")
    verilogString should include("end else if (sel == 3'h3) begin")
    result should containLine("if (sel == 3'h4) begin")
    verilogString should include("end else if (sel == 3'h5) begin")
    result should containLine("if (sel == 3'h6) begin")
    verilogString should include("end else if (sel == 3'h7) begin")
    result should containLine("tmp <= in_0;")
    result should containLine("tmp <= in_1;")
    result should containLine("tmp <= in_2;")
    result should containLine("tmp <= in_3;")
    result should containLine("tmp <= in_4;")
    result should containLine("tmp <= in_5;")
    result should containLine("tmp <= in_6;")
    result should containLine("tmp <= in_7;")
    result should containLine("tmp <= in_8;")
    result should containLine("tmp <= in_9;")
  }

  "SInt addition" should "have casts" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : SInt<4>
        |input y : SInt<4>
        |output z : SInt
        |z <= add(x, y)
        |""".stripMargin
    )
    result should containLine("assign z = $signed(x) + $signed(y);")
  }

  it should "NOT cast SInt literals" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : SInt<4>
        |output z : SInt
        |z <= add(x, SInt(-1))
        |""".stripMargin
    )
    result should containLine("assign z = $signed(x) - 4'sh1;")
  }

  it should "inline asSInt casts" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : UInt<4>
        |input y : UInt<4>
        |output z : SInt
        |node _T_1 = asSInt(x)
        |z <= add(_T_1, asSInt(y))
        |""".stripMargin
    )
    result should containLine("assign z = $signed(x) + $signed(y);")
  }

  "Verilog Emitter" should "drop asUInt casts on Clocks" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : Clock
        |input y : Clock
        |output z : UInt<1>
        |node _T_1 = asUInt(x)
        |z <= eq(_T_1, asUInt(y))
        |""".stripMargin
    )
    result should containLine("assign z = x == y;")
  }

  it should "drop asClock casts on UInts" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : UInt<1>
        |input y : UInt<1>
        |output z : Clock
        |node _T_1 = eq(x, y)
        |z <= asClock(_T_1)
        |""".stripMargin
    )
    result should containLine("assign z = x == y;")
  }

  it should "drop asUInt casts on AsyncResets" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : AsyncReset
        |input y : AsyncReset
        |output z : UInt<1>
        |node _T_1 = asUInt(x)
        |z <= eq(_T_1, asUInt(y))
        |""".stripMargin
    )
    result should containLine("assign z = x == y;")
  }

  it should "drop asAsyncReset casts on UInts" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : UInt<1>
        |input y : UInt<1>
        |output z : AsyncReset
        |node _T_1 = eq(x, y)
        |z <= asAsyncReset(_T_1)
        |""".stripMargin
    )
    result should containLine("assign z = x == y;")
  }

  it should "subtract positive literals instead of adding negative literals" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : SInt<8>
        |output z : SInt<9>
        |z <= add(x, SInt(-2))
        |""".stripMargin
    )
    result shouldNot containLine("assign z = $signed(x) + -8'sh2;")
    result should containLine("assign z = $signed(x) - 8'sh2;")
  }

  it should "subtract positive literals even with max negative literal" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : SInt<2>
        |output z : SInt<3>
        |z <= add(x, SInt(-2))
        |""".stripMargin
    )
    result shouldNot containLine("assign z = $signed(x) + -2'sh2;")
    result should containLine("assign z = $signed(x) - 3'sh2;")
  }

  it should "subtract positive literals even with max negative literal with no carryout" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : SInt<2>
        |output z : SInt<2>
        |z <= add(x, SInt(-2))
        |""".stripMargin
    )
    result shouldNot containLine("assign z = $signed(x) + -2'sh2;")
    result should containLine("wire [2:0] _GEN_0 = $signed(x) - 3'sh2;")
    result should containLine("assign z = _GEN_0[1:0];")
  }

  it should "not pad multiplication" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : UInt<2>
        |input y: UInt<4>
        |output z : UInt<6>
        |z <= mul(x, y)
        |""".stripMargin
    )
    result should containLine("assign z = x * y;")
  }

  it should "not pad division" in {
    val compiler = new VerilogCompiler
    val result = compileBody(
      """input x : UInt<4>
        |input y: UInt<2>
        |output z : UInt<4>
        |z <= div(x, y)
        |""".stripMargin
    )
    result should containLine("assign z = x / y;")
  }

  it should "correctly emit addition with a negative literal with width > 32" in {
    val result = compileBody(
      """input x : SInt<34>
        |output z : SInt<34>
        |z <= asSInt(tail(add(x, SInt<34>(-2)), 1))
        |""".stripMargin
    )
    result should containLine("assign z = $signed(x) - 34'sh2;")
  }

  it should "correctly emit conjunction with a negative literal with width > 32" in {
    val result = compileBody(
      """input x : SInt<34>
        |output z : SInt<34>
        |z <= asSInt(and(x, SInt<34>(-2)))
        |""".stripMargin
    )
    result should containLine("assign z = $signed(x) & -34'sh2;")
  }

  it should "emit FileInfo as Verilog comment" in {
    def result(info: String): CircuitState = compileBody(
      s"""input x : UInt<2>
         |output z : UInt<2>
         |z <= x @[$info]
         |""".stripMargin
    )
    result("test") should containLine("  assign z = x; // @[test]")
    // newlines currently are supposed to be escaped for both firrtl and Verilog
    // (alternatively one could emit a multi-line comment)
    result("test\\nx") should containLine("  assign z = x; // @[test\\nx]")
    // not sure why, but we are also escaping tabs
    result("test\\tx") should containLine("  assign z = x; // @[test\\tx]")
    // escaping closing square brackets is only a firrtl issue, should not be reflected in the Verilog emission
    result("test\\]") should containLine("  assign z = x; // @[test]]")
    // while firrtl allows for Unicode in the info field they should be escaped for Verilog
    result("test \uD83D\uDE0E") should containLine("  assign z = x; // @[test \\uD83D\\uDE0E]")

  }

  it should "emit repeated unary operators with parentheses" in {
    val result1 = compileBody(
      """input x : UInt<1>
        |output z : UInt<1>
        |z <= not(not(x))
        |""".stripMargin
    )
    result1 should containLine("assign z = ~(~x);")

    val result2 = compileBody(
      """input x : UInt<8>
        |output z : UInt<1>
        |z <= not(andr(x))
        |""".stripMargin
    )
    result2 should containLine("assign z = ~(&x);")
  }

  it should "remove random line with Memory and Register with some emission option" in {
    val input =
      s"""|circuit Foo:
          |  module Foo:
          |    input clock: Clock
          |    input reset: UInt<1>
          |    input in_0: UInt<1>
          |    output out: UInt<1>
          |    mem mem :
          |      data-type => UInt<1>
          |      depth => 1
          |      read-latency => 1
          |      write-latency => 1
          |      reader => r
          |      read-under-write => undefined
          |    reg tmp : UInt<1>, clock
          |    tmp <= in_0
          |    mem.r.addr <= tmp
          |    mem.r.en <= UInt<1>(0)
          |    mem.r.clk <= clock
          |    out <= mem.r.data
          |""".stripMargin
    val circuit = Seq(ToWorkingIR, ResolveKinds, InferTypes).foldLeft(parse(input)) { case (c, p) => p.run(c) }
    val state = CircuitState(
      circuit,
      LowForm,
      Seq(
        EmitCircuitAnnotation(classOf[VerilogEmitter]),
        CustomDefaultMemoryEmission(MemoryNoInit),
        CustomDefaultRegisterEmission(useInitAsPreset = true, disableRandomization = true)
      )
    )
    val result = (new VerilogEmitter).execute(state)
    (result.getEmittedCircuit.value should not).include("RANDOMIZE")
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
      DocStringAnnotation(ComponentName("a", modName), "multi\nline"),
      DocStringAnnotation(ComponentName("b", modName), "single line")
    )
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
        |  wire  d = """.stripMargin,
      """  /* multi
        |   * line
        |   */
        |  reg  e;""".stripMargin,
      """  // single line
        |  wire  f = """.stripMargin
    )
    // We don't use executeTest because we care about the spacing in the result
    val modName = ModuleName("Test", CircuitName("Test"))
    val annos = Seq(
      DocStringAnnotation(ComponentName("d", modName), "multi\nline"),
      DocStringAnnotation(ComponentName("e", modName), "multi\nline"),
      DocStringAnnotation(ComponentName("f", modName), "single line")
    )
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
    val annos = Seq(DocStringAnnotation(modName, "multi\nline"))
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
        |(* parallel_case *)
        |module Test(
        |""".stripMargin,
      """  /* line3
        |   *
        |   * line4
        |   */
        |  (* full_case *)
        |  input   a,""".stripMargin,
      """  /* line5
        |   *
        |   * line6
        |   */
        |  (* parallel_case, mark_debug *)
        |  wire  d = """.stripMargin
    )
    // We don't use executeTest because we care about the spacing in the result
    val modName = ModuleName("Test", CircuitName("Test"))
    val annos = Seq(
      DocStringAnnotation(modName, "line1"),
      DocStringAnnotation(modName, "line2"),
      AttributeAnnotation(modName, "parallel_case"),
      DocStringAnnotation(ComponentName("a", modName), "line3"),
      DocStringAnnotation(ComponentName("a", modName), "line4"),
      AttributeAnnotation(ComponentName("a", modName), "full_case"),
      DocStringAnnotation(ComponentName("d", modName), "line5"),
      DocStringAnnotation(ComponentName("d", modName), "line6"),
      AttributeAnnotation(ComponentName("d", modName), "parallel_case"),
      AttributeAnnotation(ComponentName("d", modName), "mark_debug")
    )
    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
    val output = finalState.getEmittedCircuit.value
    for (c <- check) {
      assert(output.contains(c))
    }
  }

  "Attribute annotations" should "be deduplicated" in {
    val compiler = new VerilogCompiler
    val input =
      """
        |circuit Top:
        |  module Child:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>
        |
        |    mem m:
        |      data-type => UInt<8>
        |      depth => 32
        |      reader => r
        |      writer => w
        |      read-latency => 1
        |      write-latency => 1
        |      read-under-write => new
        |
        |    m.r.clk <= clock
        |    m.r.addr <= rAddr
        |    m.r.en <= rEnable
        |    rData <= m.r.data
        |
        |    m.w.clk <= clock
        |    m.w.addr <= wAddr
        |    m.w.en <= wEnable
        |    m.w.data <= wData
        |    m.w.mask is invalid
        |
        |  module Top:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>[2]
        |
        |    inst c1 of Child
        |    c1.clock <= clock
        |    c1.rAddr <= rAddr
        |    c1.rEnable <= rEnable
        |    c1.wAddr <= wAddr
        |    c1.wData <= wData
        |    c1.wEnable <= wEnable
        |
        |    inst c2 of Child
        |    c2.clock <= clock
        |    c2.rAddr <= rAddr
        |    c2.rEnable <= rEnable
        |    c2.wAddr <= wAddr
        |    c2.wData <= wData
        |    c2.wEnable <= wEnable
        |
        |    rData[0] <= c1.rData
        |    rData[1] <= c2.rData
        |""".stripMargin
    val desc = "child module"
    val child1MRef = CircuitTarget("Top").module("Top").instOf("c1", "Child").ref("m")
    val child2MRef = CircuitTarget("Top").module("Top").instOf("c2", "Child").ref("m")
    val annos = Seq(
      AttributeAnnotation(child1MRef, desc),
      AttributeAnnotation(child2MRef, desc)
    )

    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
    val output = finalState.getEmittedCircuit.value
    assert(output.contains(s"  (* $desc *)"))
  }

  "Doc string annotations" should "be deduplicated" in {
    val compiler = new VerilogCompiler
    val input =
      """
        |circuit Top:
        |  module Child:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>
        |
        |    mem m:
        |      data-type => UInt<8>
        |      depth => 32
        |      reader => r
        |      writer => w
        |      read-latency => 1
        |      write-latency => 1
        |      read-under-write => new
        |
        |    m.r.clk <= clock
        |    m.r.addr <= rAddr
        |    m.r.en <= rEnable
        |    rData <= m.r.data
        |
        |    m.w.clk <= clock
        |    m.w.addr <= wAddr
        |    m.w.en <= wEnable
        |    m.w.data <= wData
        |    m.w.mask is invalid
        |
        |  module Top:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>[2]
        |
        |    inst c1 of Child
        |    c1.clock <= clock
        |    c1.rAddr <= rAddr
        |    c1.rEnable <= rEnable
        |    c1.wAddr <= wAddr
        |    c1.wData <= wData
        |    c1.wEnable <= wEnable
        |
        |    inst c2 of Child
        |    c2.clock <= clock
        |    c2.rAddr <= rAddr
        |    c2.rEnable <= rEnable
        |    c2.wAddr <= wAddr
        |    c2.wData <= wData
        |    c2.wEnable <= wEnable
        |
        |    rData[0] <= c1.rData
        |    rData[1] <= c2.rData
        |""".stripMargin
    val check =
      """  /* line1
        |   *
        |   * line2
        |   */
        |""".stripMargin

    val child1MRef = CircuitTarget("Top").module("Top").instOf("c1", "Child").ref("m")
    val child2MRef = CircuitTarget("Top").module("Top").instOf("c2", "Child").ref("m")
    val annos = Seq(
      DocStringAnnotation(child1MRef, "line1"),
      DocStringAnnotation(child1MRef, "line2"),
      DocStringAnnotation(child2MRef, "line1"),
      DocStringAnnotation(child2MRef, "line2")
    )

    val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
    val output = finalState.getEmittedCircuit.value
    assert(output.contains(check))
  }

  "AttributeAnnotation" should "fail dedup if not all instances have the annotation" in {
    val compiler = new VerilogCompiler
    val input =
      """
        |circuit Top:
        |  module Child:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>
        |
        |    mem m:
        |      data-type => UInt<8>
        |      depth => 32
        |      reader => r
        |      writer => w
        |      read-latency => 1
        |      write-latency => 1
        |      read-under-write => new
        |
        |    m.r.clk <= clock
        |    m.r.addr <= rAddr
        |    m.r.en <= rEnable
        |    rData <= m.r.data
        |
        |    m.w.clk <= clock
        |    m.w.addr <= wAddr
        |    m.w.en <= wEnable
        |    m.w.data <= wData
        |    m.w.mask is invalid
        |
        |  module Top:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>[2]
        |
        |    inst c1 of Child
        |    c1.clock <= clock
        |    c1.rAddr <= rAddr
        |    c1.rEnable <= rEnable
        |    c1.wAddr <= wAddr
        |    c1.wData <= wData
        |    c1.wEnable <= wEnable
        |
        |    inst c2 of Child
        |    c2.clock <= clock
        |    c2.rAddr <= rAddr
        |    c2.rEnable <= rEnable
        |    c2.wAddr <= wAddr
        |    c2.wData <= wData
        |    c2.wEnable <= wEnable
        |
        |    rData[0] <= c1.rData
        |    rData[1] <= c2.rData
        |""".stripMargin
    val desc = "child module"
    val child1MRef = CircuitTarget("Top").module("Top").instOf("c1", "Child").ref("m")
    val child2MRef = CircuitTarget("Top").module("Top").instOf("c2", "Child").ref("m")
    val annos = Seq(
      AttributeAnnotation(child1MRef, desc)
    )

    assertThrows[FirrtlUserException] {
      val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
      val output = finalState.getEmittedCircuit.value
    }
  }

  "DocStringAnnotation" should "fail dedup if not all instances share the annotation" in {
    val compiler = new VerilogCompiler
    val input =
      """
        |circuit Top:
        |  module Child:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>
        |
        |    mem m:
        |      data-type => UInt<8>
        |      depth => 32
        |      reader => r
        |      writer => w
        |      read-latency => 1
        |      write-latency => 1
        |      read-under-write => new
        |
        |    m.r.clk <= clock
        |    m.r.addr <= rAddr
        |    m.r.en <= rEnable
        |    rData <= m.r.data
        |
        |    m.w.clk <= clock
        |    m.w.addr <= wAddr
        |    m.w.en <= wEnable
        |    m.w.data <= wData
        |    m.w.mask is invalid
        |
        |  module Top:
        |    input clock : Clock
        |    input rAddr : UInt<5>
        |    input rEnable : UInt<1>
        |    input wAddr : UInt<5>
        |    input wData : UInt<8>
        |    input wEnable : UInt<1>
        |    output rData : UInt<8>[2]
        |
        |    inst c1 of Child
        |    c1.clock <= clock
        |    c1.rAddr <= rAddr
        |    c1.rEnable <= rEnable
        |    c1.wAddr <= wAddr
        |    c1.wData <= wData
        |    c1.wEnable <= wEnable
        |
        |    inst c2 of Child
        |    c2.clock <= clock
        |    c2.rAddr <= rAddr
        |    c2.rEnable <= rEnable
        |    c2.wAddr <= wAddr
        |    c2.wData <= wData
        |    c2.wEnable <= wEnable
        |
        |    rData[0] <= c1.rData
        |    rData[1] <= c2.rData
        |""".stripMargin
    val desc = "child module"
    val child1MRef = CircuitTarget("Top").module("Top").instOf("c1", "Child").ref("m")
    val child2MRef = CircuitTarget("Top").module("Top").instOf("c2", "Child").ref("m")
    val annos = Seq(
      DocStringAnnotation(child1MRef, "line1"),
      DocStringAnnotation(child2MRef, "line2"),
      DocStringAnnotation(child1MRef, "line1")
    )

    assertThrows[FirrtlUserException] {
      val finalState = compiler.compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos), Seq.empty)
      val output = finalState.getEmittedCircuit.value
    }
  }
}

class EmittedMacroSpec extends FirrtlPropSpec {
  property("User-defined macros for before/after initial should be supported") {
    val prefix = "Printf"
    val testDir = compileFirrtlTest(prefix, "/features")
    val harness = new File(testDir, s"top.cpp")
    copyResourceToFile(cppHarnessResourceName, harness)

    // define macros to print
    val cmdLineArgs = Seq(
      "+define+FIRRTL_BEFORE_INITIAL=initial begin $fwrite(32'h80000002, \"printing from FIRRTL_BEFORE_INITIAL macro\\n\"); end",
      "+define+FIRRTL_AFTER_INITIAL=initial begin $fwrite(32'h80000002, \"printing from FIRRTL_AFTER_INITIAL macro\\n\"); end"
    )

    BackendCompilationUtilities.verilogToCpp(prefix, testDir, List.empty, harness, extraCmdLineArgs = cmdLineArgs) #&&
      cppToExe(prefix, testDir) !
      loggingProcessLogger

    // check for expected print statements
    var saw_before = false
    var saw_after = false
    Process(s"./V${prefix}", testDir) !
      ProcessLogger(line => {
        line match {
          case "printing from FIRRTL_BEFORE_INITIAL macro" => saw_before = true
          case "printing from FIRRTL_AFTER_INITIAL macro"  => saw_after = true
          case _                                           => // Do Nothing
        }
      })

    assert(saw_before & saw_after)
  }
}
