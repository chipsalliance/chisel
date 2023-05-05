// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
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
  val pipelinedResetModule = Module(new PipelinedResetModule)

  pipelinedResetModule.reset := RegNext(RegNext(RegNext(reset)))

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

class PrintableScopeTester extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  out := in

  val w = Wire(UInt(8.W))
  w := 255.U

  val printableWire = cf"$w"
  val printablePort = cf"$in"
}

class AssertPrintableWireScope extends BasicTester {
  val mod = Module(new PrintableScopeTester)
  assert(1.U === 2.U, mod.printableWire)
  stop()
}

class AssertPrintablePortScope extends BasicTester {
  val mod = Module(new PrintableScopeTester)
  mod.in := 255.U
  assert(1.U === 1.U, mod.printablePort)
  stop()
}

class AssertPrintableFailingWhenScope extends BasicTester {
  val mod = Module(new PrintableWhenScopeTester)
  assert(1.U === 1.U, mod.printable)
  stop()
}

class AssumePrintableWireScope extends BasicTester {
  val mod = Module(new PrintableScopeTester)
  assume(1.U === 1.U, mod.printableWire)
  stop()
}

class AssumePrintablePortScope extends BasicTester {
  val mod = Module(new PrintableScopeTester)
  mod.in := 255.U
  assume(1.U === 1.U, mod.printablePort)
  stop()
}

class PrintableWhenScopeTester extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))

  out := in

  val w = Wire(UInt(8.W))
  w := 255.U
  var printable = cf""
  when(true.B) {
    printable = cf"$w"
  }
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

  "Assert Printables" should "respect port scoping" in {
    assertTesterPasses { new AssertPrintablePortScope }
  }
  "Assert Printables" should "respect wire scoping" in {
    a[ChiselException] should be thrownBy { ChiselStage.emitCHIRRTL(new AssertPrintableWireScope) }
  }
  "Assume Printables" should "respect port scoping" in {
    assertTesterPasses { new AssumePrintablePortScope }
  }

  "Assume Printables" should "respect wire scoping" in {
    a[ChiselException] should be thrownBy { ChiselStage.emitCHIRRTL(new AssumePrintableWireScope) }
  }

  "Assert Printables" should "respect when scope" in {
    a[ChiselException] should be thrownBy { ChiselStage.emitCHIRRTL(new AssertPrintableFailingWhenScope) }
  }

  "Assertions" should "allow the modulo operator % in the message" in {
    assertTesterPasses { new ModuloAssertTester }
  }
  they should "allow printf-style format strings with arguments" in {
    assertTesterPasses { new FormattedAssertTester }
  }
  they should "allow printf-style format strings in Assumes" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new PrintableAssumeTester)
    chirrtl should include(
      """assume(w === 255.U, cf\"Assumption failed, Wire w =/= $w%%%%x\")\n", w)"""
    )
  }
  they should "not allow unescaped % in the message" in {
    a[java.util.UnknownFormatConversionException] should be thrownBy {
      extractCause[java.util.UnknownFormatConversionException] {
        ChiselStage.emitCHIRRTL { new BadUnescapedPercentAssertTester }
      }
    }
  }

  they should "allow printable format strings with arguments" in {
    assertTesterPasses { new FormattedAssertTester }
  }
  they should "not allow unescaped % in the printable message" in {
    a[java.util.UnknownFormatConversionException] should be thrownBy {
      extractCause[java.util.UnknownFormatConversionException] {
        ChiselStage.emitCHIRRTL { new BadUnescapedPercentAssertTester }
      }
    }
  }

}
