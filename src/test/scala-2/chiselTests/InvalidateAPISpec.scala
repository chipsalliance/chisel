// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util.Counter
import firrtl.util.BackendCompilationUtilities._
import circt.stage.ChiselStage
import org.scalatest._
import org.scalatest.matchers.should.Matchers

class InvalidateAPISpec extends ChiselPropSpec with Matchers with Utils {

  def myGenerateFirrtl(t: => Module): String = ChiselStage.emitCHIRRTL(t)
  def compileFirrtl(t: => Module): Unit = {
    val testDir = createTestDirectory(this.getClass.getSimpleName)

    (new ChiselStage).execute(
      Array[String]("-td", testDir.getAbsolutePath, "--target", "verilog"),
      Seq(ChiselGeneratorAnnotation(() => t))
    )
  }
  class TrivialInterface extends Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
  }

  property("an output connected to DontCare should emit a Firrtl \"invalidate\"") {
    class ModuleWithDontCare extends Module {
      val io = IO(new TrivialInterface)
      io.out := DontCare
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithDontCare)
    firrtlOutput should include("invalidate io.out")
  }

  property("an output without a DontCare should NOT emit a Firrtl \"invalidate\"") {
    class ModuleWithoutDontCare extends Module {
      val io = IO(new TrivialInterface)
      io.out := io.in
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    (firrtlOutput should not).include("invalidate")
  }

  property("a bundle with a DontCare should emit a Firrtl \"invalidate\"") {
    class ModuleWithoutDontCare extends Module {
      val io = IO(new TrivialInterface)
      io <> DontCare
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    firrtlOutput should include("invalidate io.out")
    firrtlOutput should include("invalidate io.in")
  }

  property("a Vec with a DontCare should emit a Firrtl \"invalidate\" with bulk connect") {
    val nElements = 5
    class ModuleWithoutDontCare extends Module {
      val io = IO(new Bundle {
        val outs = Output(Vec(nElements, Bool()))
      })
      io.outs <> DontCare
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    for (i <- 0 until nElements)
      firrtlOutput should include(s"invalidate io.outs[$i]")
  }

  property("a Vec with a DontCare should emit a Firrtl \"invalidate\" with mono connect") {
    val nElements = 5
    class ModuleWithoutDontCare extends Module {
      val io = IO(new Bundle {
        val ins = Input(Vec(nElements, Bool()))
      })
      io.ins := DontCare
    }
    val firrtlOutput = myGenerateFirrtl(new ModuleWithoutDontCare)
    for (i <- 0 until nElements)
      firrtlOutput should include(s"invalidate io.ins[$i]")
  }

  property("a DontCare cannot be a connection sink (LHS) for := ") {
    class ModuleWithDontCareSink extends Module {
      val io = IO(new TrivialInterface)
      DontCare := io.in
    }
    val exception = intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL(new ModuleWithDontCareSink)
      }
    }
    exception.getMessage should include("DontCare cannot be a connection sink")
  }

  property("a DontCare cannot be a connection sink (LHS) for <>") {
    class ModuleWithDontCareSink extends Module {
      val io = IO(new TrivialInterface)
      DontCare <> io.in
    }
    val exception = intercept[BiConnectException] {
      extractCause[BiConnectException] {
        circt.stage.ChiselStage.emitCHIRRTL(new ModuleWithDontCareSink)
      }
    }
    exception.getMessage should include("DontCare cannot be a connection sink (LHS)")
  }

  property("FIRRTL should complain about partial initialization with conditional connect") {
    class ModuleWithIncompleteAssignment extends Module {
      val io = IO(new Bundle {
        val out = Output(Bool())
      })
      val counter = Counter(8)
      when(counter.inc()) {
        io.out := true.B
      }
    }
    val exception = intercept[RuntimeException] {
      circt.stage.ChiselStage.emitSystemVerilog(new ModuleWithIncompleteAssignment)
    }
    exception.getMessage should include("not fully initialized")
  }

  property(
    "FIRRTL should not complain about partial initialization with conditional connect after unconditional connect"
  ) {
    class ModuleWithUnconditionalAssignment extends Module {
      val io = IO(new Bundle {
        val out = Output(Bool())
      })
      val counter = Counter(8)
      io.out := false.B
      when(counter.inc()) {
        io.out := true.B
      }
    }
    circt.stage.ChiselStage.emitSystemVerilog(new ModuleWithUnconditionalAssignment)
  }

  property(
    "FIRRTL should not complain about partial initialization with conditional connect with otherwise clause"
  ) {
    class ModuleWithConditionalAndOtherwiseAssignment extends Module {
      val io = IO(new Bundle {
        val out = Output(Bool())
      })
      val counter = Counter(8)
      when(counter.inc()) {
        io.out := true.B
      }.otherwise {
        io.out := false.B
      }
    }

    circt.stage.ChiselStage.emitSystemVerilog(new ModuleWithConditionalAndOtherwiseAssignment)
  }

  property("a clock should be able to be connected to a DontCare") {
    class ClockConnectedToDontCare extends Module {
      val foo = IO(Output(Clock()))
      foo := DontCare
    }
    myGenerateFirrtl(new ClockConnectedToDontCare) should include("invalidate foo")
  }
}
