// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.passes.memlib.SeparateWriteClocks
import firrtl.testutils._
import firrtl.testutils.FirrtlCheckers._

class SeparateWriteClocksSpec extends FirrtlFlatSpec {
  def transform(input: String): CircuitState = {
    val csx = (new SeparateWriteClocks).execute(CircuitState(parse(input), MidForm))
    val emittedCirc = EmittedFirrtlCircuit("top", csx.circuit.serialize, ".fir")
    csx.copy(annotations = Seq(EmittedFirrtlCircuitAnnotation(emittedCirc)))
  }

  behavior.of("SeparateWriteClocks")

  it should "add intermediate wires to clocks of multi-write sync-read memories" in {
    val result = transform(s"""
                              |circuit top:
                              |  module top:
                              |    input clk: Clock
                              |    input raddr: UInt<10>
                              |    output rdata: UInt<8>[4]
                              |    input waddr_a: UInt<10>
                              |    input we_a: UInt<1>
                              |    input wdata_a: UInt<8>[4]
                              |    input waddr_a: UInt<10>
                              |    input we_a: UInt<1>
                              |    input wdata_a: UInt<8>[4]
                              |
                              |    mem m:
                              |      data-type => UInt<8>
                              |      depth => 1024
                              |      reader => r
                              |      writer => w_a
                              |      writer => w_b
                              |      read-latency => 1
                              |      write-latency => 1
                              |      read-under-write => undefined
                              |
                              |    m.r.clk <= clk
                              |    m.r.addr <= raddr
                              |    m.r.en <= UInt(1)
                              |    rdata <= m.r.data
                              |
                              |    m.w_a.clk <= clk
                              |    m.w_a.addr <= waddr_a
                              |    m.w_a.en <= we_a
                              |    m.w_a.mask <= UInt(1)
                              |    m.w_a.data <= wdata_a
                              |
                              |    m.w_b.clk <= clk
                              |    m.w_b.addr <= waddr_b
                              |    m.w_b.en <= we_b
                              |    m.w_b.mask <= UInt(1)
                              |    m.w_b.data <= wdata_b""".stripMargin)

    result should containLine("m.r.clk <= clk")
    result should containLine("m.w_a.clk <= m_w_a_clk")
    result should containLine("m.w_b.clk <= m_w_b_clk")
    result shouldNot containLine("m.w_a.clk <= clk")
    result shouldNot containLine("m.w_b.clk <= clk")
  }
}
