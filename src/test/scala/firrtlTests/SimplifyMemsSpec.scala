// See LICENSE for license details.

package firrtlTests

import firrtl.transforms._
import firrtl._

import CompilerUtils.getLoweringTransforms

class SimplifyMemsSpec extends ConstantPropagationSpec {
  override val transforms = getLoweringTransforms(ChirrtlForm, MidForm) ++ Seq(new SimplifyMems)

  "SimplifyMems" should "lower aggregate memories" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input wen : UInt<1>
        |    input wdata : { a : UInt<8>, b : UInt<8> }
        |    output rdata : { a : UInt<8>, b : UInt<8> }
        |    mem m :
        |      data-type => { a : UInt<8>, b : UInt<8>}
        |      depth => 32
        |      read-latency => 1
        |      write-latency => 1
        |      reader => read
        |      writer => write
        |    m.read.clk <= clock
        |    m.read.en <= UInt<1>(1)
        |    m.read.addr is invalid
        |    rdata <= m.read.data
        |    m.write.clk <= clock
        |    m.write.en <= wen
        |    m.write.mask.a <= UInt<1>(1)
        |    m.write.mask.b <= UInt<1>(1)
        |    m.write.addr is invalid
        |    m.write.data <= wdata

     """.stripMargin

    val check =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input wen : UInt<1>
        |    input wdata : { a : UInt<8>, b : UInt<8>}
        |    output rdata : { a : UInt<8>, b : UInt<8>}
        |
        |    wire m : { flip read : { addr : UInt<5>, en : UInt<1>, clk : Clock, flip data : { a : UInt<8>, b : UInt<8>}}, flip write : { addr : UInt<5>, en : UInt<1>, clk : Clock, data : { a : UInt<8>, b : UInt<8>}, mask : { a : UInt<1>, b : UInt<1>}}}
        |    mem m_flattened :
        |      data-type => UInt<16>
        |      depth => 32
        |      read-latency => 1
        |      write-latency => 1
        |      reader => read
        |      writer => write
        |      read-under-write => undefined
        |    m_flattened.read.addr <= m.read.addr
        |    m_flattened.read.en <= m.read.en
        |    m_flattened.read.clk <= m.read.clk
        |    m.read.data.b <= asUInt(bits(m_flattened.read.data, 7, 0))
        |    m.read.data.a <= asUInt(bits(m_flattened.read.data, 15, 8))
        |    m_flattened.write.addr <= m.write.addr
        |    m_flattened.write.en <= m.write.en
        |    m_flattened.write.clk <= m.write.clk
        |    m_flattened.write.data <= cat(asUInt(m.write.data.a), asUInt(m.write.data.b))
        |    m_flattened.write.mask <= UInt<1>("h1")
        |    rdata.a <= m.read.data.a
        |    rdata.b <= m.read.data.b
        |    m.read.addr is invalid
        |    m.read.en <= UInt<1>("h1")
        |    m.read.clk <= clock
        |    m.write.addr is invalid
        |    m.write.en <= wen
        |    m.write.clk <= clock
        |    m.write.data.a <= wdata.a
        |    m.write.data.b <= wdata.b
        |    m.write.mask.a <= UInt<1>("h1")
        |    m.write.mask.b <= UInt<1>("h1")

     """.stripMargin
    (parse(exec(input))) should be(parse(check))
  }
}
