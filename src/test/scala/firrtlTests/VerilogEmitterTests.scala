// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl._
import firrtl.annotations._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo

class DoPrimVerilog extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], compiler: Compiler) = {
    val writer = new StringWriter()
    compiler.compile(CircuitState(parse(input), ChirrtlForm), writer)
    val lines = writer.toString().split("\n") map normalized
    expected foreach { e =>
      lines should contain(e)
    }
  }
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
}
