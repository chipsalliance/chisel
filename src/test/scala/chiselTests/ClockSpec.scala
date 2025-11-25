// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import circt.stage.ChiselStage
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers

class ClockAsUIntTester extends Module {
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

class ClockSpec extends AnyPropSpec with Matchers with ChiselSim {
  property("Bool.asClock.asUInt should pass a signal through unaltered") {
    simulate { new ClockAsUIntTester }(RunUntilFinished(3))
  }

  property("Should be able to use withClock in a module with no reset") {
    val circuit = ChiselStage.emitCHIRRTL(new WithClockAndNoReset)
    circuit.contains("reg a : UInt<1>, clock2") should be(true)
  }

  property("Should be able to override the value of the implicit clock") {
    val verilog = ChiselStage.emitSystemVerilog(new Module {
      val gate = IO(Input(Bool()))
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val gatedClock = (clock.asBool || gate).asClock
      override protected def implicitClock = gatedClock

      val r = Reg(UInt(8.W))
      out := r
      r := in
    })
    verilog should include("gatedClock = clock | gate;")
    verilog should include("always @(posedge gatedClock)")
  }

  property("Should be able to add an implicit clock to a RawModule") {
    val verilog = ChiselStage.emitSystemVerilog(new RawModule with ImplicitClock {
      val foo = IO(Input(Bool()))
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      override protected val implicitClock = (!foo).asClock

      val r = Reg(UInt(8.W))
      out := r
      r := in
    })
    verilog should include("always @(posedge implicitClock)")
  }

  property("Chisel should give a decent error message if you try to use a clock before defining it") {
    val e = the[ChiselException] thrownBy (
      ChiselStage.emitCHIRRTL(
        new RawModule with ImplicitClock {
          val r = Reg(UInt(8.W))
          val foo = IO(Input(Clock()))
          override protected def implicitClock = foo
        },
        args = Array("--throw-on-first-error")
      )
    )
    e.getMessage should include(
      "The implicit clock is null which means the code that sets its definition has not yet executed."
    )
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
      "operand 'Child.clock: Wire[Clock]' is not visible from the current module Parent"
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
