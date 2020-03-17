// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.testutils.FirrtlFlatSpec

// Tests long-standing bug from #1179, VerilogMemDelays producing combinational loops in corner case
abstract class MemEnFeedbackSpec extends FirrtlFlatSpec {
  val ruw: String
  def input: String =
    s"""circuit loop :
       |  module loop :
       |    input clk : Clock
       |    input raddr : UInt<5>
       |    mem m :
       |      data-type => UInt<1>
       |      depth => 32
       |      reader => r
       |      read-latency => 1
       |      write-latency => 1
       |      read-under-write => ${ruw}
       |    m.r.clk <= clk
       |    m.r.addr <= raddr
       |    m.r.en <= m.r.data
       |""".stripMargin
  def compileInput(): Unit = (new VerilogCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), List.empty)
}

class WriteFirstMemEnFeedbackSpec extends MemEnFeedbackSpec {
  val ruw = "new"
  "A write-first sync-read mem with feedback from data to enable" should "compile without errors" in {
    compileInput()
  }
}

class ReadFirstMemEnFeedbackSpec extends MemEnFeedbackSpec {
  val ruw = "old"
  "A read-first sync-read mem with feedback from data to enable" should "compile without errors" in {
    compileInput()
  }
}
