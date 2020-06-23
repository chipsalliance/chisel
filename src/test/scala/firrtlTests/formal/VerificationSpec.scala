package firrtlTests.formal

import firrtl.{SystemVerilogCompiler}
import firrtl.testutils.FirrtlFlatSpec
import logger.{LogLevel, Logger}

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
        |    if (1'h1) begin
        |      assume(inputEquals0xAA);
        |    end
        |    if (1'h1) begin
        |      assert(areEqual);
        |    end
        |    if (1'h1) begin
        |      cover(outputEquals0xAA);
        |    end
        |  end
        |endmodule
        |""".stripMargin.split("\n") map normalized
    executeTest(input, expected, compiler)
  }
}
