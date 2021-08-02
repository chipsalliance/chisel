// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.testutils._
import firrtl.ir.Circuit
import firrtl.options.Dependency
import firrtl.passes.memlib.VerilogMemDelays

class VerilogMemDelaySpec extends LeanTransformSpec(Seq(Dependency(VerilogMemDelays))) {
  behavior.of("VerilogMemDelaySpec")

  private def compileTwiceReturnFirst(input: String): Circuit = {
    val res0 = compile(input)
    compile(res0.circuit.serialize).circuit
  }

  private def compileTwice(input: String): Unit = compileTwiceReturnFirst(input)

  it should "The following low FIRRTL should be parsed by VerilogMemDelays" in {
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

  it should "Using a read-first memory should be allowed in VerilogMemDelays" in {
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

  it should "Chained memories should generate correct FIRRTL" in {
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

  it should "VerilogMemDelays should not violate use before declaration of clocks" in {
    val input =
      """
        |circuit Test :
        |  extmodule ClockMaker :
        |    input in : UInt<8>
        |    output clock : Clock
        |  module Test :
        |    input clock : Clock
        |    input addr : UInt<5>
        |    input mask : UInt<8>
        |    input in : UInt<8>
        |    output out : UInt<8>
        |    mem m :
        |      data-type => UInt<8>
        |      depth => 32
        |      read-latency => 1
        |      write-latency => 2
        |      reader => read
        |      writer => write
        |      read-under-write => old  ; this is important
        |    inst cm of ClockMaker
        |    m.read.clk <= cm.clock
        |    m.read.en <= UInt<1>(1)
        |    m.read.addr <= addr
        |    out <= m.read.data
        |    ; This makes this really funky for injected pipe ordering
        |    node read = not(m.read.data)
        |    cm.in <= and(read, UInt<8>("hf0"))
        |
        |    m.write.clk <= clock
        |    m.write.en <= UInt<1>(1)
        |    m.write.mask <= mask
        |    m.write.addr <= addr
        |    m.write.data <= in
      """.stripMargin

    val res = compile(input).circuit.serialize
    // Inject a Wire when using a clock not derived from ports
    res should include("wire m_clock : Clock")
    res should include("m_clock <= cm.clock")
    res should include("reg m_read_data_pipe_0 : UInt<8>, m_clock")
    res should include("m.read.clk <= m_clock")
    // No need to insert Wire when clock is derived from a port
    res should include("m.write.clk <= clock")
    res should include("reg m_write_data_pipe_0 : UInt<8>, clock")
  }
}
