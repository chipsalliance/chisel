// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.util._

class FailingAssertTester() extends BasicTester {
  assert(false.B)
  // Wait to come out of reset
  val (_, done) = Counter(!reset.asBool, 4)
  when(done) {
    stop()
  }
}

class SucceedingAssertTester() extends BasicTester {
  assert(true.B)
  // Wait to come out of reset
  val (_, done) = Counter(!reset.asBool, 4)
  when(done) {
    stop()
  }
}

class PipelinedResetModule extends Module {
  val io = IO(new Bundle {})
  val a = RegInit(0xbeef.U)
  val b = RegInit(0xbeef.U)
  assert(a === b)
}

// This relies on reset being asserted for 3 or more cycles
class PipelinedResetTester extends BasicTester {
  val module = Module(new PipelinedResetModule)

  module.reset := RegNext(RegNext(RegNext(reset)))

  val (_, done) = Counter(!reset.asBool, 4)
  when(done) {
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

class PrintableFormattedAssertTester extends BasicTester {
  val foobar = Wire(UInt(32.W))
  foobar := 123.U
  assert(foobar === 123.U, cf"Error! Wire foobar =/= $foobar%x This is 100%% wrong.\n")
  stop()
}

class PrintableBadUnescapedPercentAssertTester extends BasicTester {
  assert(1.U === 1.U, p"I'm 110% sure this is an invalid message")
  stop()
}

class PrintableAssumeTester extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))

  val w = Wire(UInt(8.W))
  w := 255.U
  assume(w === 255.U, cf"Assumption failed, Wire w =/= $w%x")

  out := in
}

class AssertSpec extends ChiselFlatSpec with Utils {
  "A failing assertion" should "fail the testbench" in {
    assert(!runTester { new FailingAssertTester })
  }
  "A succeeding assertion" should "not fail the testbench" in {
    assertTesterPasses { new SucceedingAssertTester }
  }
  "An assertion" should "not assert until we come out of reset" in {
    assertTesterPasses { new PipelinedResetTester }
  }
  "Assertions" should "allow the modulo operator % in the message" in {
    assertTesterPasses { new ModuloAssertTester }
  }
  they should "allow printf-style format strings with arguments" in {
    assertTesterPasses { new FormattedAssertTester }
  }
  they should "allow printf-style format strings in Assumes" in {
    val chirrtl = ChiselStage.emitChirrtl(new PrintableAssumeTester)
    chirrtl should include(
      """assume(w === 255.U, cf\"Assumption failed, Wire w =/= $w%%%%x\")\n", w)"""
    )
  }
  they should "not allow unescaped % in the message" in {
    a[java.util.UnknownFormatConversionException] should be thrownBy {
      extractCause[java.util.UnknownFormatConversionException] {
        ChiselStage.elaborate { new BadUnescapedPercentAssertTester }
      }
    }
  }

  they should "allow printable format strings with arguments" in {
    assertTesterPasses { new FormattedAssertTester }
  }
  they should "not allow unescaped % in the printable message" in {
    a[java.util.UnknownFormatConversionException] should be thrownBy {
      extractCause[java.util.UnknownFormatConversionException] {
        ChiselStage.elaborate { new BadUnescapedPercentAssertTester }
      }
    }
  }

}
