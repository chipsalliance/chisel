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

  case class SignedMuxTest(
    operatorFIRRTL:  String,
    operatorVerilog: String,
    inputWidth:      Int,
    outputWidth:     Int,
    secondArgSigned: Boolean) {
    private val moduleName = s"Signed${operatorFIRRTL.capitalize}Mux"
    private val headerFIRRTL =
      s"""|circuit $moduleName :
          |  module $moduleName :
          |    input sel : UInt<1>
          |    input is0 : SInt<$inputWidth>
          |    input is1 : ${if (secondArgSigned) "S" else "U"}Int<$inputWidth>
          |    output os : SInt<$outputWidth>""".stripMargin
    private val expressionFIRRTL = s"$operatorFIRRTL(is0, is1)"
    private val headerVerilog =
      s"""|module Reference(
          |  input sel,
          |  input signed [$inputWidth-1:0] is0,
          |  input ${if (secondArgSigned) "signed" else ""}[$inputWidth-1:0] is1,
          |  output signed [$outputWidth-1:0] os
          |);""".stripMargin
    private val expressionVerilog = s"is0 $operatorVerilog is1"

    def firrtlWhen: String =
      s"""|$headerFIRRTL
          |    os <= SInt(0)
          |    when sel :
          |      os <= $expressionFIRRTL
          |""".stripMargin

    def firrtlTrue: String =
      s"""|$headerFIRRTL
          |    os <= mux(sel, $expressionFIRRTL, SInt(0))
          |""".stripMargin

    def firrtlFalse: String =
      s"""|$headerFIRRTL
          |    os <= mux(sel, SInt(0), $expressionFIRRTL)
          |""".stripMargin

    def verilogTrue: String =
      s"""|$headerVerilog
          |  assign os = sel ? $expressionVerilog : $outputWidth'sh0;
          |endmodule
          |""".stripMargin
    def verilogFalse: String =
      s"""|$headerVerilog
          |  assign os = sel ? $outputWidth'sh0 : $expressionVerilog;
          |endmodule
          |""".stripMargin
  }

  Seq(
    SignedMuxTest("add", "+", 2, 3, true),
    SignedMuxTest("sub", "-", 2, 3, true),
    SignedMuxTest("mul", "*", 2, 4, true),
    SignedMuxTest("div", "/", 2, 3, true),
    SignedMuxTest("rem", "%", 2, 2, true),
    SignedMuxTest("dshl", "<<", 2, 3, false)
  ).foreach {
    case t =>
      s"signed ${t.operatorFIRRTL} followed by a mux" should "be correct" in {
        info(s"'when' where '${t.operatorFIRRTL}' used in the 'when' body okay")
        firrtlEquivalenceWithVerilog(t.firrtlWhen, t.verilogTrue)
        info(s"'mux' where '${t.operatorFIRRTL}' used in the true leg okay")
        firrtlEquivalenceWithVerilog(t.firrtlTrue, t.verilogTrue)
        info(s"'mux' where '${t.operatorFIRRTL}' used in the false leg okay")
        firrtlEquivalenceWithVerilog(t.firrtlFalse, t.verilogFalse)
      }
  }

  "signed shl followed by a mux" should "be correct" in {
    val headerFIRRTL =
      """|circuit SignedShlMux :
         |  module SignedShlMux :
         |    input sel : UInt<1>
         |    input is0 : SInt<2>
         |    output os : SInt<5>""".stripMargin
    val headerVerilog =
      """|module Reference(
         |  input sel,
         |  input signed [1:0] is0,
         |  output signed [4:0] os
         |);""".stripMargin

    val firrtlWhen =
      s"""|$headerFIRRTL
          |    os <= SInt(0)
          |    when sel :
          |      os <= shl(is0, 2)
          |""".stripMargin

    val firrtlTrue =
      s"""|$headerFIRRTL
          |    os <= mux(sel, shl(is0, 2), SInt(0))
          |""".stripMargin

    val firrtlFalse =
      s"""|$headerFIRRTL
          |    os <= mux(sel, SInt(0), shl(is0, 2))
          |""".stripMargin

    val verilogTrue =
      s"""|$headerVerilog
          |  assign os = sel ? (is0 << 2'h2) : 5'sh0;
          |endmodule
          |""".stripMargin

    val verilogFalse =
      s"""|$headerVerilog
          |  assign os = sel ? 5'sh0 : (is0 << 2'h2);
          |endmodule
          |""".stripMargin

    firrtlEquivalenceWithVerilog(firrtlWhen, verilogTrue)
    firrtlEquivalenceWithVerilog(firrtlTrue, verilogTrue)
    firrtlEquivalenceWithVerilog(firrtlFalse, verilogFalse)
  }

}
