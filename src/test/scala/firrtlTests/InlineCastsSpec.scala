// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.transforms.InlineCastsTransform
import firrtl.testutils.FirrtlFlatSpec
import firrtl.testutils.FirrtlCheckers._

class InlineCastsEquivalenceSpec extends FirrtlFlatSpec {
  /*
   * Note: InlineCasts is still part of mverilog, so this test must both:
   * - Test that the InlineCasts fix is effective given the current mverilog
   * - Provide a test that will be robust if and when InlineCasts is no longer run in mverilog
   *
   * This is why the test passes InlineCasts as a custom transform: to future-proof it so that
   * it can do real LEC against no-InlineCasts. It currently is just a sanity check that the
   * emitted Verilog is legal, but it will automatically become a more meaningful test when
   * InlineCasts is not run in mverilog.
   */
  "InlineCastsTransform" should "not produce broken Verilog" in {
    val input =
      s"""circuit literalsel_fir:
         |  module literalsel_fir:
         |    input i: UInt<4>
         |    output o: SInt<8>
         |    o <= pad(asSInt(UInt<2>("h1")), 8)
         |""".stripMargin
    firrtlEquivalenceTest(input, Seq(new InlineCastsTransform))
  }

  it should "not inline complex expressions into other complex expressions" in {
    val input =
      """circuit NeverInlineComplexIntoComplex :
        |  module NeverInlineComplexIntoComplex :
        |    input a : SInt<3>
        |    input b : UInt<2>
        |    input c : UInt<2>
        |    input sel : UInt<1>
        |    output out : SInt<3>
        |    node diff = sub(b, c)
        |    out <= mux(sel, a, asSInt(diff))
        |""".stripMargin
    val expected =
      """module NeverInlineComplexIntoComplexRef(
        |  input  [2:0] a,
        |  input  [1:0] b,
        |  input  [1:0] c,
        |  input        sel,
        |  output [2:0] out
        |);
        |  wire [2:0] diff = b - c;
        |  assign out = sel ? $signed(a) : $signed(diff);
        |endmodule
        |""".stripMargin
    firrtlEquivalenceWithVerilog(input, expected)
  }

  it should "inline casts on both sides of a more complex expression" in {
    val input =
      """circuit test :
        |  module test :
        |    input clock : Clock
        |    input in : UInt<8>
        |    output out : UInt<8>
        |
        |    node _T_1 = asUInt(clock)
        |    node _T_2 = not(_T_1)
        |    node clock_n = asClock(_T_2)
        |    reg r : UInt<8>, clock_n
        |    r <= in
        |    out <= r
        |""".stripMargin
    val verilog = compileToVerilogCircuitState(input)
    verilog should containLine("always @(posedge clock_n) begin")

  }
}
