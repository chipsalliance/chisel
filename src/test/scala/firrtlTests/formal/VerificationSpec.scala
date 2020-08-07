// See LICENSE for license details.

package firrtlTests.formal

import firrtl.{CircuitState, SystemVerilogCompiler, ir}
import firrtl.testutils.FirrtlFlatSpec
import logger.{LogLevel, Logger}
import firrtl.options.Dependency
import firrtl.stage.TransformManager

class VerificationSpec extends FirrtlFlatSpec {
  behavior of "Formal"

  it should "generate SystemVerilog verification statements" in {
    val compiler = new SystemVerilogCompiler
    val input =
      """circuit Asserting :
        |  module Asserting :
        |    input clock: Clock
        |    input reset: UInt<1>
        |    input in: UInt<8>
        |    output out: UInt<8>
        |    wire areEqual: UInt<1>
        |    wire inputEquals0xAA: UInt<1>
        |    wire outputEquals0xAA: UInt<1>
        |    out <= in
        |    areEqual <= eq(out, in)
        |    inputEquals0xAA <= eq(in, UInt<8>("hAA"))
        |    outputEquals0xAA <= eq(out, UInt<8>("hAA"))
        |    node true = UInt("b1")
        |    assume(clock, inputEquals0xAA, true, "assume input is 0xAA")
        |    assert(clock, areEqual, true, "assert that output equals input")
        |    cover(clock, outputEquals0xAA, true, "cover output is 0xAA")
        |""".stripMargin
    val expected =
      """module Asserting(
        |  input [7:0] in,
        |  output [7:0] out
        |);
        |  wire areEqual = out == in;
        |  wire inputEquals0xAA = in == 8'haa;
        |  wire outputEquals0xAA = out == 8'haa;
        |  assign out = in;
        |  always @(posedge clock) begin
        |    // assume input is 0xAA
        |    if (1'h1) begin
        |      assume(inputEquals0xAA);
        |    end
        |    // assert that output equals input
        |    if (1'h1) begin
        |      assert(areEqual);
        |    end
        |    // cover output is 0xAA
        |    if (1'h1) begin
        |      cover(outputEquals0xAA);
        |    end
        |  end
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, expected, compiler)
  }

  "VerificationStatement" should "serialize correctly" in {
    val clk = ir.Reference("clk")
    val en = ir.Reference("en")
    val pred = ir.Reference("pred")
    val a = ir.Verification(ir.Formal.Assert, ir.NoInfo, clk, pred, en, ir.StringLit("test"))

    assert(a.serialize == "assert(clk, pred, en, \"test\")")
    assert(ir.Serializer.serialize(a) == "assert(clk, pred, en, \"test\")")

    val b = ir.Verification(ir.Formal.Assume, ir.NoInfo, clk, en, pred, ir.StringLit("test \n test"))
    assert(b.serialize == "assume(clk, en, pred, \"test \\n test\")")
    assert(ir.Serializer.serialize(b) == "assume(clk, en, pred, \"test \\n test\")")

    val c = ir.Verification(ir.Formal.Assume, ir.NoInfo, clk, pred, en, ir.StringLit("test \t test"))
    assert(c.serialize == "assume(clk, pred, en, \"test \\t test\")")
    assert(ir.Serializer.serialize(c) == "assume(clk, pred, en, \"test \\t test\")")

  }

  "VerificationStatements" should "end up at the bottom of the circuit like other simulation statements" in {
    val compiler = new TransformManager(Seq(Dependency(firrtl.passes.ExpandWhens)))
    val in =
      """circuit m :
        |  module m :
        |    input clock : Clock
        |    input a : UInt<8>
        |    output b : UInt<16>
        |    b <= a
        |    assert(clock, eq(a, b), UInt<1>("h1"), "")
        |""".stripMargin
    val afterExpandWhens = compiler.transform(CircuitState(firrtl.Parser.parse(in), Seq())).circuit.serialize
    val lastLine = afterExpandWhens.split("\n").last
    assert(lastLine.trim.startsWith("assert"))
  }
}
