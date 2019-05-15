// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.RawModule
import chisel3.testers.BasicTester

class ClockAsUIntTester extends BasicTester {
  assert(true.B.asClock.asUInt === 1.U)
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
    val circuit = Driver.emit { () => new WithClockAndNoReset }
    circuit.contains("reg a : UInt<1>, clock2") should be (true)
  }
}
