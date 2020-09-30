// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.passes.memlib.VerilogMemDelays
import firrtl.passes.CheckHighForm
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlSourceAnnotation, FirrtlStage}

import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class VerilogMemDelaySpec extends AnyFreeSpec with Matchers {

  private def compileTwice(input: String): Unit = {
    val result1 = (new FirrtlStage).transform(Seq(FirrtlSourceAnnotation(input))).toSeq.collectFirst {
      case fca: FirrtlCircuitAnnotation => (new FirrtlStage).transform(Seq(fca))
    }
  }

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

    compileTwice(input)
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

    compileTwice(input)
  }

  "Chained memories should generate correct FIRRTL" in {
    val input =
      """
        |circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input addr : UInt<5>
        |    input wdata : UInt<32>
        |    input wmode : UInt<1>
        |    output rdata : UInt<32>
        |    mem m1 :
        |      data-type => UInt<32>
        |      depth => 32
        |      read-latency => 1
        |      write-latency => 1
        |      read-under-write => old
        |      readwriter => rw
        |    m1.rw.clk <= clock
        |    m1.rw.en <= UInt<1>(1)
        |    m1.rw.addr <= addr
        |    m1.rw.wmode <= wmode
        |    m1.rw.wmask <= UInt<1>(1)
        |    m1.rw.wdata <= wdata
        |
        |    mem m2 :
        |      data-type => UInt<32>
        |      depth => 32
        |      read-latency => 1
        |      write-latency => 1
        |      read-under-write => old
        |      readwriter => rw
        |    m2.rw.clk <= clock
        |    m2.rw.en <= UInt<1>(1)
        |    m2.rw.addr <= addr
        |    m2.rw.wmode <= wmode
        |    m2.rw.wmask <= UInt<1>(1)
        |    m2.rw.wdata <= m1.rw.rdata
        |
        |    rdata <= m2.rw.rdata
        |""".stripMargin

    compileTwice(input)
  }
}
