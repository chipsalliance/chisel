// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.Counter
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FailingAssertTester() extends Module {
  assert(false.B)
  // Wait to come out of reset
  val (_, done) = Counter(!reset.asBool, 4)
  when(done) {
    stop()
  }
}

class SucceedingAssertTester() extends Module {
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
class PipelinedResetTester extends Module {
  val pipelinedResetModule = Module(new PipelinedResetModule)

  pipelinedResetModule.reset := RegNext(RegNext(RegNext(reset)))

  val (_, done) = Counter(!reset.asBool, 4)
  when(done) {
    stop()
  }
}

class ModuloAssertTester extends Module {
  assert((4.U % 2.U) === 0.U)
  stop()
}

class FormattedAssertTester extends Module {
  val foobar = Wire(UInt(32.W))
  foobar := 123.U
  assert(foobar === 123.U, "Error! Wire foobar =/= %x! This is 100%% wrong.\n", foobar)
  stop()
}

class BadUnescapedPercentAssertTester extends Module {
  assert(1.U === 1.U, "I'm 110% sure this is an invalid message")
  stop()
}

class PrintableFormattedAssertTester extends Module {
  val foobar = Wire(UInt(32.W))
  foobar := 123.U
  assert(foobar === 123.U, cf"Error! Wire foobar =/= $foobar%x This is 100%% wrong.\n")
  stop()
}

class PrintableBadUnescapedPercentAssertTester extends Module {
  assert(1.U === 1.U, p"I'm 110% sure this is an invalid message")
  stop()
}

class PrintableAssumeTester extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))

  val w = Wire(UInt(8.W))
  w := 255.U
  assume(w === 255.U, cf"wire w =/= $w%x")

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

class AssertPrintableWireScope extends Module {
  val mod = Module(new PrintableScopeTester)
  assert(1.U === 2.U, mod.printableWire)
  stop()
}

class AssertPrintablePortScope extends Module {
  val mod = Module(new PrintableScopeTester)
  mod.in := 255.U
  assert(1.U === 1.U, mod.printablePort)
  stop()
}

class AssertPrintableFailingWhenScope extends Module {
  val mod = Module(new PrintableWhenScopeTester)
  assert(1.U === 1.U, mod.printable)
  stop()
}

class AssumePrintableWireScope extends Module {
  val mod = Module(new PrintableScopeTester)
  assume(1.U === 1.U, mod.printableWire)
  stop()
}

class AssumePrintablePortScope extends Module {
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

class AssertSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "A failing assertion" should "fail the testbench" in {
    intercept[Exception] {
      simulate(new FailingAssertTester)(RunUntilFinished(3))
    }.getMessage() should include("One or more assertions failed")
  }
  "A succeeding assertion" should "not fail the testbench" in {
    simulate { new SucceedingAssertTester }(RunUntilFinished(5))
  }
  "An assertion" should "not assert until we come out of reset" in {
    simulate { new PipelinedResetTester }(RunUntilFinished(5))
  }

  "Assert Printables" should "respect port scoping" in {
    simulate { new AssertPrintablePortScope }(RunUntilFinished(3))
  }
  "Assert Printables" should "respect wire scoping" in {
    a[ChiselException] should be thrownBy { ChiselStage.emitCHIRRTL(new AssertPrintableWireScope) }
  }
  "Assume Printables" should "respect port scoping" in {
    simulate { new AssumePrintablePortScope }(RunUntilFinished(3))
  }

  "Assume Printables" should "respect wire scoping" in {
    a[ChiselException] should be thrownBy { ChiselStage.emitCHIRRTL(new AssumePrintableWireScope) }
  }

  "Assert Printables" should "respect when scope" in {
    a[ChiselException] should be thrownBy { ChiselStage.emitCHIRRTL(new AssertPrintableFailingWhenScope) }
  }

  "Assertions" should "allow the modulo operator % in the message" in {
    simulate { new ModuloAssertTester }(RunUntilFinished(3))
  }
  they should "allow printf-style format strings with arguments" in {
    simulate { new FormattedAssertTester }(RunUntilFinished(3))
  }
  they should "allow printf-style format strings in Assumes" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new PrintableAssumeTester)
    chirrtl should include(
      """"Assumption failed: wire w =/= %x\n", w"""
    )
  }
  they should "not allow unescaped % in the message" in {
    intercept[java.util.UnknownFormatConversionException] {
      ChiselStage.emitCHIRRTL { new BadUnescapedPercentAssertTester }
    }
  }

  they should "allow printable format strings with arguments" in {
    simulate { new FormattedAssertTester }(RunUntilFinished(3))
  }
  they should "not allow unescaped % in the printable message" in {
    intercept[java.util.UnknownFormatConversionException] {
      ChiselStage.emitCHIRRTL { new BadUnescapedPercentAssertTester }
    }
  }

}
