// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class FailingAssertTester() extends BasicTester {
  assert(false.B)
  // Wait to come out of reset
  val (_, done) = Counter(!reset.toBool, 4)
  when (done) {
    stop()
  }
}

class SucceedingAssertTester() extends BasicTester {
  assert(true.B)
  // Wait to come out of reset
  val (_, done) = Counter(!reset.toBool, 4)
  when (done) {
    stop()
  }
}

class PipelinedResetModule extends Module {
  val io = IO(new Bundle { })
  val a = RegInit(0xbeef.U)
  val b = RegInit(0xbeef.U)
  assert(a === b)
}

// This relies on reset being asserted for 3 or more cycles
class PipelinedResetTester extends BasicTester {
  val module = Module(new PipelinedResetModule)

  module.reset := RegNext(RegNext(RegNext(reset)))

  val (_, done) = Counter(!reset.toBool, 4)
  when (done) {
    stop()
  }
}

class ModuloAssertTester extends BasicTester {
  assert((4.U % 2.U) === 0.U)
  stop()
}

class FormattedAssertTester extends BasicTester {
  val foobar = Wire(UInt(32.W))
  foobar := 123.U
  assert(foobar === 123.U, "Error! Wire foobar =/= %x! This is 100%% wrong.\n", foobar)
  stop()
}

class BadUnescapedPercentAssertTester extends BasicTester {
  assert(1.U === 1.U, "I'm 110% sure this is an invalid message")
  stop()
}

class AssertSpec extends ChiselFlatSpec {
  "A failing assertion" should "fail the testbench" in {
    assert(!runTester{ new FailingAssertTester })
  }
  "A succeeding assertion" should "not fail the testbench" in {
    assertTesterPasses{ new SucceedingAssertTester }
  }
  "An assertion" should "not assert until we come out of reset" in {
    assertTesterPasses{ new PipelinedResetTester }
  }
  "Assertions" should "allow the modulo operator % in the message" in {
    assertTesterPasses{ new ModuloAssertTester }
  }
  they should "allow printf-style format strings with arguments" in {
    assertTesterPasses{ new FormattedAssertTester }
  }
  they should "not allow unescaped % in the message" in {
    a [java.util.UnknownFormatConversionException] should be thrownBy {
      elaborate { new BadUnescapedPercentAssertTester }
    }
  }
}
