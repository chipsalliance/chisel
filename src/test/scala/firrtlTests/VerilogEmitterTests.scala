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
}
