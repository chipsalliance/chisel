// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.testutils._

class NegSpec extends FirrtlFlatSpec {
  "unsigned neg" should "be correct and lint-clean" in {
    val input =
      """|circuit UnsignedNeg :
         |  module UnsignedNeg  :
         |    input in : UInt<8>
         |    output out : SInt
         |    out <= neg(in)
         |""".stripMargin
    val expected =
      """|module UnsignedNegRef(
         |  input [7:0] in,
         |  output [8:0] out
         |);
         |  assign out = 8'd0 - in;
         |endmodule""".stripMargin
    firrtlEquivalenceWithVerilog(input, expected)
    lintVerilog(compileToVerilog(input))
  }

  "signed neg" should "be correct and lint-clean" in {
    val input =
      """|circuit SignedNeg :
         |  module SignedNeg  :
         |    input in : SInt<8>
         |    output out : SInt
         |    out <= neg(in)
         |""".stripMargin
    // -$signed(in) is a lint warning in Verilator but is functionally correct
    val expected =
      """|module SignedNegRef(
         |  input [7:0] in,
         |  output [8:0] out
         |);
         |  assign out = -$signed(in);
         |endmodule""".stripMargin
    firrtlEquivalenceWithVerilog(input, expected)
    lintVerilog(compileToVerilog(input))
  }
}
