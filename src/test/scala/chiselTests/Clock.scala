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
}
