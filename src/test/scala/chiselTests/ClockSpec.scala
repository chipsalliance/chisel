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

  property("Chisel should give a decent error message if you use an unbound Clock") {
    val e = the[ChiselException] thrownBy (
      ChiselStage.emitCHIRRTL(
        new RawModule {
          withClock(Clock()) {
            val r = Reg(UInt(8.W))
          }
        },
        args = Array("--throw-on-first-error")
      )
    )
    e.getMessage should include(
      "'Clock' must be hardware, not a bare Chisel type. Perhaps you forgot to wrap it in Wire(_) or IO(_)?"
    )
  }

  property("Chisel should give a decent error message if you use a Clock from another scope") {
    val e = the[ChiselException] thrownBy (
      ChiselStage.emitCHIRRTL(
        new RawModule {
          override def desiredName = "Parent"
          val child = Module(new RawModule {
            override def desiredName = "Child"
            val clock = Wire(Clock())
          })
          withClock(child.clock) {
            val r = Reg(UInt(8.W))
          }
        },
        args = Array("--throw-on-first-error")
      )
    )
    e.getMessage should include(
      "operand 'Child.clock: Wire[Clock]' is not visible from the current module"
    )
  }

  property("Chisel should support Clocks from views") {
    import chisel3.experimental.dataview._
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val clock = IO(Clock())
      val view = clock.viewAs[Clock]
      withClock(view) {
        val r = Reg(UInt(8.W))
      }
    })
    chirrtl should include("reg r : UInt<8>, clock")
  }
}
