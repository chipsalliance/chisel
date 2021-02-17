// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.testutils._

class VerilogEquivalenceSpec extends FirrtlFlatSpec {
  "mul followed by cat" should "be correct" in {
    val header = s"""
                    |circuit Multiply :
                    |  module Multiply :
                    |    input x : UInt<4>
                    |    input y : UInt<2>
                    |    input z : UInt<2>
                    |    output out : UInt<8>
                    |""".stripMargin
    val input1 = header + """
                            |    out <= cat(z, mul(x, y))""".stripMargin
    val input2 = header + """
                            |    node n = mul(x, y)
                            |    node m = cat(z, n)
                            |    out <= m""".stripMargin
    val expected = s"""
                      |module MultiplyRef(
                      |  input [3:0] x,
                      |  input [1:0] y,
                      |  input [1:0] z,
                      |  output [7:0] out
                      |);
                      |  wire [5:0] w = x * y;
                      |  assign out = {z, w};
                      |endmodule""".stripMargin
    firrtlEquivalenceWithVerilog(input1, expected)
    firrtlEquivalenceWithVerilog(input2, expected)
  }

  "div followed by cat" should "be correct" in {
    val header = s"""
                    |circuit Divide :
                    |  module Divide :
                    |    input x : UInt<4>
                    |    input y : UInt<2>
                    |    input z : UInt<2>
                    |    output out : UInt<6>
                    |""".stripMargin
    val input1 = header + """
                            |    out <= cat(z, div(x, y))""".stripMargin
    val input2 = header + """
                            |    node n = div(x, y)
                            |    node m = cat(z, n)
                            |    out <= m""".stripMargin
    val expected = s"""
                      |module DivideRef(
                      |  input [3:0] x,
                      |  input [1:0] y,
                      |  input [1:0] z,
                      |  output [5:0] out
                      |);
                      |  wire [3:0] w = x / y;
                      |  assign out = {z, w};
                      |endmodule""".stripMargin
    firrtlEquivalenceWithVerilog(input1, expected)
    firrtlEquivalenceWithVerilog(input2, expected)
  }

  "signed mul followed by cat" should "be correct" in {
    val header = s"""
                    |circuit SignedMultiply :
                    |  module SignedMultiply :
                    |    input x : SInt<4>
                    |    input y : SInt<2>
                    |    input z : SInt<2>
                    |    output out : UInt<8>
                    |""".stripMargin
    val input1 = header + """
                            |    out <= cat(z, mul(x, y))""".stripMargin
    val input2 = header + """
                            |    node n = mul(x, y)
                            |    node m = cat(z, n)
                            |    out <= m""".stripMargin
    val expected = s"""
                      |module SignedMultiplyRef(
                      |  input signed [3:0] x,
                      |  input signed [1:0] y,
                      |  input signed [1:0] z,
                      |  output [7:0] out
                      |);
                      |  wire [5:0] w = x * y;
                      |  assign out = {z, w};
                      |endmodule""".stripMargin
    firrtlEquivalenceWithVerilog(input1, expected)
    firrtlEquivalenceWithVerilog(input2, expected)
  }

  "signed div followed by cat" should "be correct" in {
    val header = s"""
                    |circuit SignedDivide :
                    |  module SignedDivide :
                    |    input x : SInt<4>
                    |    input y : SInt<2>
                    |    input z : SInt<2>
                    |    output out : UInt<7>
                    |""".stripMargin
    val input1 = header + """
                            |    out <= cat(z, div(x, y))""".stripMargin
    val input2 = header + """
                            |    node n = div(x, y)
                            |    node m = cat(z, n)
                            |    out <= m""".stripMargin
    val expected = s"""
                      |module SignedDivideRef(
                      |  input signed [3:0] x,
                      |  input signed [1:0] y,
                      |  input signed [1:0] z,
                      |  output [6:0] out
                      |);
                      |  wire [4:0] w = x / y;
                      |  assign out = {z, w};
                      |endmodule""".stripMargin
    firrtlEquivalenceWithVerilog(input1, expected)
    firrtlEquivalenceWithVerilog(input2, expected)
  }
}
