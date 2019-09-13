// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.passes.memlib.VerilogMemDelays
import firrtl.passes.CheckHighForm
import org.scalatest.{FreeSpec, Matchers}

class VerilogMemDelaySpec extends FreeSpec with Matchers {
  "The following low FIRRTL should be parsed by VerilogMemDelays" in {
    val input =
      """
        |circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input addr : UInt<5>
        |    input mask : { a : UInt<1>, b: UInt<1> }[2]
        |    output out : { a : UInt<8>, b : UInt<8>}[2]
        |    mem m :
        |      data-type => { a : UInt<8>, b : UInt<8>}[2]
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 1
        |      reader => read
        |      writer => write
        |    m.read.clk <= clock
        |    m.read.en <= UInt<1>(1)
        |    m.read.addr <= addr
        |    out <= m.read.data
        |
        |    m.write.clk <= clock
        |    m.write.en <= UInt<1>(1)
        |    m.write.mask <= mask
        |    m.write.addr <= addr
        |    wire w : { a : UInt<8>, b : UInt<8>}[2]
        |    w[0].a <= UInt<4>(2)
        |    w[0].b <= UInt<4>(3)
        |    w[1].a <= UInt<4>(4)
        |    w[1].b <= UInt<4>(5)
        |    m.write.data <= w
      """.stripMargin

    val circuit = Parser.parse(input)
    val compiler = new LowFirrtlCompiler

    val result = compiler.compile(CircuitState(circuit, ChirrtlForm), Seq.empty)
    val result2 = VerilogMemDelays.run(result.circuit)
    CheckHighForm.run(result2)
    //result.circuit.serialize.length > 0 should be (true)
  }

  "Using a read-first memory should be allowed in VerilogMemDelays" in {
    val input =
      """
        |circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input waddr : UInt<5>
        |    input wdata : UInt<32>
        |    input raddr : UInt<5>
        |    input rw_wen : UInt<1>
        |    output rdata : UInt<32>
        |    output rw_rdata : UInt<32>
        |    mem m :
        |      data-type => UInt<32>
        |      depth => 32
        |      read-latency => 1
        |      write-latency => 1
        |      read-under-write => old
        |      reader => read
        |      writer => write
        |      readwriter => rw
        |    m.read.clk <= clock
        |    m.read.en <= UInt<1>(1)
        |    m.read.addr <= raddr
        |    rdata <= m.read.data
        |
        |    m.write.clk <= clock
        |    m.write.en <= UInt<1>(1)
        |    m.write.mask <= UInt<1>(1)
        |    m.write.addr <= waddr
        |    m.write.data <= wdata
        |
        |    m.rw.clk <= clock
        |    m.rw.en <= UInt<1>(1)
        |    m.rw.wmode <= rw_wen
        |    m.rw.wmask <= UInt<1>(1)
        |    m.rw.addr <= waddr
        |    m.rw.wdata <= wdata
        |    rw_rdata <= m.rw.rdata
      """.stripMargin

    val circuit = Parser.parse(input)
    val compiler = new LowFirrtlCompiler

    val result = compiler.compile(CircuitState(circuit, ChirrtlForm), Seq.empty)
    val result2 = VerilogMemDelays.run(result.circuit)
    CheckHighForm.run(result2)
  }
}
