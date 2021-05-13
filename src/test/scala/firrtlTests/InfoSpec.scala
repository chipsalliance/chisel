// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.testutils._
import FirrtlCheckers._
import firrtl.Parser.AppendInfo

class InfoSpec extends FirrtlFlatSpec with FirrtlMatchers {
  def compile(input: String): CircuitState =
    (new VerilogCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), List.empty)
  def compileBody(body: String) = {
    val str = """
                |circuit Test :
                |  module Test :
                |""".stripMargin + body.split("\n").mkString("    ", "\n    ", "")
    compile(str)
  }

  // Some useful constants to use and look for
  val Info1 = FileInfo(StringLit("Source.scala 1:4"))
  val Info2 = FileInfo(StringLit("Source.scala 2:4"))
  val Info3 = FileInfo(StringLit("Source.scala 3:4"))
  val Info4 = FileInfo(StringLit("Source.scala 4:4"))

  "Source locators on module ports" should "be propagated to Verilog" in {
    val result = compileBody(s"""
                                |input x : UInt<8> $Info1
                                |output y : UInt<8> $Info2
                                |y <= x""".stripMargin)
    result should containTree { case Port(Info1, "x", Input, _) => true }
    result should containLine(s"input [7:0] x, //$Info1")
    result should containTree { case Port(Info2, "y", Output, _) => true }
    result should containLine(s"output [7:0] y //$Info2")
  }

  "Source locators on aggregates" should "be propagated to Verilog" in {
    val result = compileBody(s"""
                                |input io : { x : UInt<8>, flip y : UInt<8> } $Info1
                                |io.y <= io.x""".stripMargin)
    result should containTree { case Port(Info1, "io_x", Input, _) => true }
    result should containLine(s"input [7:0] io_x, //$Info1")
    result should containTree { case Port(Info1, "io_y", Output, _) => true }
    result should containLine(s"output [7:0] io_y //$Info1")
  }

  "Source locators" should "be propagated on declarations" in {
    val result = compileBody(s"""
                                |input clock : Clock
                                |input x : UInt<8>
                                |output y : UInt<8>
                                |reg r : UInt<8>, clock $Info1
                                |wire w : UInt<8> $Info2
                                |node n = or(w, x) $Info3
                                |w <= and(x, r)
                                |r <= or(n, r)
                                |y <= r""".stripMargin)
    result should containTree { case DefRegister(Info1, "r", _, _, _, _) => true }
    result should containLine(s"reg [7:0] r; //$Info1")
    result should containTree { case DefNode(Info2, "w", _) => true }
    result should containLine(s"wire [7:0] w = x & r; //$Info2") // Node "w" declaration in Verilog
    result should containTree { case DefNode(Info3, "n", _) => true }
    result should containLine(s"wire [7:0] n = w | x; //$Info3")
  }

  it should "be propagated on memories" in {
    val result = compileBody(s"""
                                |input clock : Clock
                                |input addr : UInt<5>
                                |output z : UInt<8>
                                |mem m: $Info1
                                |  data-type => UInt<8>
                                |  depth => 32
                                |  read-latency => 0
                                |  write-latency => 1
                                |  reader => r
                                |  writer => w
                                |m.r.clk <= clock
                                |m.r.addr <= addr
                                |m.r.en <= UInt(1)
                                |m.w.clk <= clock
                                |m.w.addr <= addr
                                |m.w.en <= UInt(0)
                                |m.w.data <= UInt(0)
                                |m.w.mask <= UInt(0)
                                |z <= m.r.data
                                |""".stripMargin)

    result should containTree { case DefMemory(Info1, "m", _, _, _, _, _, _, _, _) => true }
    result should containLine(s"reg [7:0] m [0:31]; //$Info1")
    result should containLine(s"wire [7:0] m_r_data; //$Info1")
    result should containLine(s"wire [4:0] m_r_addr; //$Info1")
    result should containLine(s"wire [7:0] m_w_data; //$Info1")
    result should containLine(s"wire [4:0] m_w_addr; //$Info1")
    result should containLine(s"wire  m_w_mask; //$Info1")
    result should containLine(s"wire  m_w_en; //$Info1")
    result should containLine(s"assign m_r_data = m[m_r_addr]; //$Info1")
    result should containLine(s"m[m_w_addr] <= m_w_data; //$Info1")
  }

  it should "be propagated on instances" in {
    val result = compile(s"""
                            |circuit Test :
                            |  module Child :
                            |    output io : { flip in : UInt<8>, out : UInt<8> }
                            |    io.out <= io.in
                            |  module Test :
                            |    output io : { flip in : UInt<8>, out : UInt<8> }
                            |    inst c of Child $Info1
                            |    io <= c.io
                            |""".stripMargin)
    result should containTree { case WDefInstance(Info1, "c", "Child", _) => true }
    result should containLine(s"Child c ( //$Info1")
  }

  it should "be propagated across direct node assignments and connections" in {
    val result = compile(s"""
                            |circuit Test :
                            |  module Test :
                            |    input in : UInt<8>
                            |    output out : UInt<8>
                            |    node a = in $Info1
                            |    node b = a
                            |    out <= b
                            |""".stripMargin)
    result should containTree { case Connect(Info1, Reference("out", _, _, _), Reference("in", _, _, _)) => true }
    result should containLine(s"assign out = in; //$Info1")
  }

  "source locators" should "be propagated through ExpandWhens" in {
    val input =
      """
        |;buildInfoPackage: chisel3, version: 3.1-SNAPSHOT, scalaVersion: 2.11.7, sbtVersion: 0.13.11, builtAtString: 2016-11-26 18:48:38.030, builtAtMillis: 1480186118030
        |circuit GCD :
        |  module GCD :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output io : {flip a : UInt<32>, flip b : UInt<32>, flip e : UInt<1>, z : UInt<32>, v : UInt<1>}
        |
        |    io is invalid
        |    io is invalid
        |    reg x : UInt<32>, clock @[GCD.scala 15:14]
        |    reg y : UInt<32>, clock @[GCD.scala 16:14]
        |    node _T_14 = gt(x, y) @[GCD.scala 17:11]
        |    when _T_14 : @[GCD.scala 17:18]
        |      node _T_15 = sub(x, y) @[GCD.scala 17:27]
        |      node _T_16 = tail(_T_15, 1) @[GCD.scala 17:27]
        |      x <= _T_16 @[GCD.scala 17:22]
        |      skip @[GCD.scala 17:18]
        |    node _T_18 = eq(_T_14, UInt<1>("h00")) @[GCD.scala 17:18]
        |    when _T_18 : @[GCD.scala 18:18]
        |      node _T_19 = sub(y, x) @[GCD.scala 18:27]
        |      node _T_20 = tail(_T_19, 1) @[GCD.scala 18:27]
        |      y <= _T_20 @[GCD.scala 18:22]
        |      skip @[GCD.scala 18:18]
        |    when io.e : @[GCD.scala 19:15]
        |      x <= io.a @[GCD.scala 19:19]
        |      y <= io.b @[GCD.scala 19:30]
        |      skip @[GCD.scala 19:15]
        |    io.z <= x @[GCD.scala 20:8]
        |    node _T_22 = eq(y, UInt<1>("h00")) @[GCD.scala 21:13]
        |    io.v <= _T_22 @[GCD.scala 21:8]
        |
      """.stripMargin

    val result = (new LowFirrtlCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), List.empty)
    result should containLine("node _GEN_0 = mux(_T_14, _T_16, x) @[GCD.scala 15:14 17:{18,22}]")
    result should containLine("node _GEN_2 = mux(io_e, io_a, _GEN_0) @[GCD.scala 19:{15,19}]")
    result should containLine("x <= _GEN_2")
    result should containLine("node _GEN_1 = mux(_T_18, _T_20, y) @[GCD.scala 16:14 18:{18,22}]")
    result should containLine("node _GEN_3 = mux(io_e, io_b, _GEN_1) @[GCD.scala 19:{15,30}]")
    result should containLine("y <= _GEN_3")
  }

  "source locators for append option" should "use multiinfo" in {
    val input = """circuit Top :
                  |  module Top :
                  |    input clock : Clock
                  |    input in: UInt<32>
                  |    output out: UInt<32>
                  |    out <= in @[Top.scala 15:14]
                  |""".stripMargin
    val circuit = firrtl.Parser.parse(input.split("\n").toIterator, AppendInfo("myfile.fir"))
    val circuitState = CircuitState(circuit, UnknownForm)
    val expectedInfos = Seq(FileInfo(StringLit("Top.scala 15:14")), FileInfo(StringLit("myfile.fir 6:4")))
    circuitState should containTree { case MultiInfo(`expectedInfos`) => true }
  }

  "source locators for basic register updates" should "be propagated to Verilog" in {
    val result = compileBody(s"""
                                |input clock : Clock
                                |input reset : UInt<1>
                                |output io : { flip in : UInt<8>, out : UInt<8>}
                                |reg r : UInt<8>, clock
                                |r <= io.in $Info1
                                |io.out <= r
                                |""".stripMargin)
    result should containLine(s"r <= io_in; //$Info1")
  }

  "source locators for register reset" should "be propagated to Verilog" in {
    val result = compileBody(s"""
                                |input clock : Clock
                                |input reset : UInt<1>
                                |output io : { flip in : UInt<8>, out : UInt<8>}
                                |reg r : UInt<8>, clock with : (reset => (reset, UInt<8>("h0"))) $Info3
                                |r <= io.in $Info1
                                |io.out <= r
                                |""".stripMargin)
    result should containLine(s"if (reset) begin //$Info3")
    result should containLine(s"r <= 8'h0; //$Info3")
    result should containLine(s"r <= io_in; //$Info1")
  }

  "source locators for complex register updates" should "be propagated to Verilog" in {
    val result = compileBody(s"""
                                |input clock : Clock
                                |input reset : UInt<1>
                                |output io : { flip in : UInt<8>, flip a : UInt<1>, out : UInt<8>}
                                |reg r : UInt<8>, clock with : (reset => (reset, UInt<8>("h0"))) $Info1
                                |r <= UInt<2>(2) $Info2
                                |when io.a : $Info3
                                |  r <= io.in $Info4
                                |io.out <= r
                                |""".stripMargin)
    result should containLine(s"if (reset) begin //$Info1")
    result should containLine(s"r <= 8'h0; //$Info1")
    result should containLine(s"end else if (io_a) begin //$Info3")
    result should containLine(s"r <= io_in; //$Info4")
    result should containLine(s"r <= 8'h2; //$Info2")
  }

  "FileInfo" should "be able to contain a escaped characters" in {
    def input(info: String): String =
      s"""circuit m: @[$info]
         |  module m:
         |    skip
         |""".stripMargin
    def parseInfo(info: String): FileInfo = {
      firrtl.Parser.parse(input(info)).info.asInstanceOf[FileInfo]
    }

    parseInfo("test\\ntest").escaped should be("test\\ntest")
    parseInfo("test\\ntest").unescaped should be("test\ntest")
    parseInfo("test\\ttest").escaped should be("test\\ttest")
    parseInfo("test\\ttest").unescaped should be("test\ttest")
    parseInfo("test\\\\test").escaped should be("test\\\\test")
    parseInfo("test\\\\test").unescaped should be("test\\test")
    parseInfo("test\\]test").escaped should be("test\\]test")
    parseInfo("test\\]test").unescaped should be("test]test")
    parseInfo("test[\\][\\]test").escaped should be("test[\\][\\]test")
    parseInfo("test[\\][\\]test").unescaped should be("test[][]test")
  }

  it should "be compressed in Verilog whenever possible" in {
    def result(info1: String, info2: String, info3: String) = compileBody(
      s"""output out:UInt<32>
         |input b:UInt<32>
         |input c:UInt<1>
         |input d:UInt<32>
         |wire a:UInt<32>
         |when c : @[$info1]
         |  a <= b @[$info2]
         |else :
         |  a <= d @[$info3]
         |out <= add(a,a)""".stripMargin
    )

    // Keep different file infos separated
    result("A 1:1", "B 1:1", "C 1:1") should containLine("  wire [31:0] a = c ? b : d; // @[A 1:1 B 1:1 C 1:1]")
    // Compress only 2 FileInfos of the same file
    result("A 1:1", "A 2:3", "C 1:1") should containLine("  wire [31:0] a = c ? b : d; // @[A 1:1 2:3 C 1:1]")
    // Conmpress 3 lines from the same file into one single FileInfo
    result("A 1:2", "A 2:4", "A 3:6") should containLine("  wire [31:0] a = c ? b : d; // @[A 1:2 2:4 3:6]")
    // Compress two columns from the same line, and one different line into one FileInfo
    result("A 1:2", "A 1:4", "A 2:3") should containLine("  wire [31:0] a = c ? b : d; // @[A 1:{2,4} 2:3]")
    // Compress three (or more...) columns from the same line into one FileInfo
    result("A 1:2", "A 1:3", "A 1:4") should containLine("  wire [31:0] a = c ? b : d; // @[A 1:{2,3,4}]")

    // Ignore already-compressed MultiInfos - for when someone may serialize a module first and compile the parsed firrtl into Verilog
    result("A 1:{2,3,4}", "", "") should containLine(
      "  wire [31:0] a = c ? b : d; // @[A 1:{2,3,4}]"
    )
    // Merge additional FileInfos together, but ignore compressed MultiInfos if they are present
    result("A 1:{2,3,4}", "B 2:3", "B 4:5") should containLine(
      "  wire [31:0] a = c ? b : d; // @[A 1:{2,3,4} B 2:3 4:5]"
    )
    result("A 2:3", "B 1:{2,3,4}", "C 4:5") should containLine(
      "  wire [31:0] a = c ? b : d; // @[B 1:{2,3,4} A 2:3 C 4:5]"
    )
  }

  it should "not be compressed if it has a non-conforming format" in {
    // Sample module from the firrtl spec for file info comments
    val result = compileBody(
      """output out:UInt @["myfile.txt: 16, 3"]
        |input b:UInt<32> @["myfile.txt: 17, 3"]
        |input c:UInt<1> @["myfile.txt: 18, 3"]
        |input d:UInt<16> @["myfile.txt: 19, 3"]
        |wire a:UInt @["myfile.txt: 21, 8"]
        |when c : @["myfile.txt: 24, 8"]
        |  a <= b @["myfile.txt: 27, 16"]
        |else :
        |  a <= d @["myfile.txt: 29, 17"]
        |out <= add(a,a) @["myfile.txt: 34, 4"]
        |""".stripMargin
    )

    // Should compile to the following lines in the test module
    val check = Seq(
      """  output [32:0] out, // @[\"myfile.txt: 16, 3\"]""",
      """  input  [31:0] b, // @[\"myfile.txt: 17, 3\"]""",
      """  input         c, // @[\"myfile.txt: 18, 3\"]""",
      """  input  [15:0] d // @[\"myfile.txt: 19, 3\"]""",
      """  wire [31:0] a = c ? b : {{16'd0}, d}; // @[\"myfile.txt: 24, 8\" \"myfile.txt: 27, 16\" \"myfile.txt: 29, 17\"]""",
      """  assign out = a + a; // @[\"myfile.txt: 34, 4\"]"""
    )

    for (line <- check)
      result should containLine(line)
  }
}
