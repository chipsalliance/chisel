// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import FirrtlCheckers._

class InfoSpec extends FirrtlFlatSpec {
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

  "Source locators on module ports" should "be propagated to Verilog" in {
    val result = compileBody(s"""
      |input x : UInt<8> $Info1
      |output y : UInt<8> $Info2
      |y <= x""".stripMargin
    )
    result should containTree { case Port(Info1, "x", Input, _) => true }
    result should containLine (s"input [7:0] x, //$Info1")
    result should containTree { case Port(Info2, "y", Output, _) => true }
    result should containLine (s"output [7:0] y //$Info2")
  }

  "Source locators on aggregates" should "be propagated to Verilog" in {
    val result = compileBody(s"""
      |input io : { x : UInt<8>, flip y : UInt<8> } $Info1
      |io.y <= io.x""".stripMargin
    )
    result should containTree { case Port(Info1, "io_x", Input, _) => true }
    result should containLine (s"input [7:0] io_x, //$Info1")
    result should containTree { case Port(Info1, "io_y", Output, _) => true }
    result should containLine (s"output [7:0] io_y //$Info1")
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
      |y <= r""".stripMargin
    )
    result should containTree { case DefRegister(Info1, "r", _,_,_,_) => true }
    result should containLine (s"reg [7:0] r; //$Info1")
    result should containTree { case DefNode(Info2, "w", _) => true }
    result should containLine (s"wire [7:0] w; //$Info2") // Node "w" declaration in Verilog
    result should containTree { case DefNode(Info3, "n", _) => true }
    result should containLine (s"wire [7:0] n; //$Info3")
    result should containLine (s"assign n = w | x; //$Info3")
  }

  they should "be propagated on memories" in {
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
      |""".stripMargin
    )

    result should containTree { case DefMemory(Info1, "m", _,_,_,_,_,_,_,_) => true }
    result should containLine (s"reg [7:0] m [0:31]; //$Info1")
    result should containLine (s"wire [7:0] m_r_data; //$Info1")
    result should containLine (s"wire [4:0] m_r_addr; //$Info1")
    result should containLine (s"wire [7:0] m_w_data; //$Info1")
    result should containLine (s"wire [4:0] m_w_addr; //$Info1")
    result should containLine (s"wire  m_w_mask; //$Info1")
    result should containLine (s"wire  m_w_en; //$Info1")
    result should containLine (s"assign m_r_data = m[m_r_addr]; //$Info1")
    result should containLine (s"m[m_w_addr] <= m_w_data; //$Info1")
  }

  they should "be propagated on instances" in {
    val result = compile(s"""
      |circuit Test :
      |  module Child :
      |    output io : { flip in : UInt<8>, out : UInt<8> }
      |    io.out <= io.in
      |  module Test :
      |    output io : { flip in : UInt<8>, out : UInt<8> }
      |    inst c of Child $Info1
      |    io <= c.io
      |""".stripMargin
    )
    result should containTree { case WDefInstance(Info1, "c", "Child", _) => true }
    result should containLine (s"Child c ( //$Info1")
  }
}
