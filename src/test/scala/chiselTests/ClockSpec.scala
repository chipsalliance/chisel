// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import circt.stage.ChiselStage

class ClockAsUIntTester extends BasicTester {
  assert(true.B.asClock.asUInt === 1.U)
  assert(true.B.asClock.asBool === true.B)
  stop()
}

class WithClockAndNoReset extends RawModule {
  val clock1 = IO(Input(Clock()))
  val clock2 = IO(Input(Clock()))
  val in = IO(Input(Bool()))
  val out = IO(Output(Bool()))
  val a = withClock(clock2) {
    RegNext(in)
  }

  out := a
}

class ClockSpec extends ChiselPropSpec {
  property("Bool.asClock.asUInt should pass a signal through unaltered") {
    assertTesterPasses { new ClockAsUIntTester }
  }

  property("Should be able to use withClock in a module with no reset") {
    val circuit = ChiselStage.emitCHIRRTL(new WithClockAndNoReset)
    circuit.contains("reg a : UInt<1>, clock2") should be(true)
  }

  property("Should be able to override the value of the implicit clock") {
    val verilog = ChiselStage.emitSystemVerilog(new Module with OverrideClock {
      val gate = IO(Input(Bool()))
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val gatedClock = (clock.asBool || gate).asClock
      internalClock := gatedClock
      val r = Reg(UInt(8.W))
      out := r
      r := in
    })
    // Signal name really should be gatedClock, hopefully fixed in future version of firtool
    verilog should include("_gatedClock_T_2 = clock | gate;")
    verilog should include("always @(posedge _gatedClock_T_2)")
  }
}
