package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.experimental.withClock
import chisel3.util._
import chisel3.tester._

class ClockCrossingTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2 with clock crossing signals"

  it should "test crossing from a 2:1 divider domain" in {
    test(new Module {
      val io = IO(new Bundle {
        val divIn = Input(UInt(8.W))
        val mainOut = Output(UInt(8.W))
      })

      val divClock = RegInit(true.B)
      divClock := !divClock

      val divRegWire = Wire(UInt())
      withClock(divClock.asClock) {
        val divReg = RegNext(io.divIn, 1.U)
        divRegWire := divReg
      }

      val mainReg = RegNext(divRegWire, 0.U)
      io.mainOut := mainReg
    }) { c =>
      c.io.divIn.poke(0x42.U)
      c.io.mainOut.expect(0.U)  // initial register value
      c.clock.step(1)
      c.io.mainOut.expect(1.U)  // initial value of divReg
      c.clock.step(1)  // for divided clock to have a rising edge
      c.io.mainOut.expect(1.U)  // one-cycle-delaye divReg
      c.clock.step(1)  // for main clock register to propagate
      c.io.mainOut.expect(0x42.U)  // updated value propagates
    }
  }
}
