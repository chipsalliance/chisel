package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.experimental.withClock
import chisel3.util._
import chisel3.tester._

class ClockDividerTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2 with a clock divider"

  it should "test 1:2 clock divider counter" in {
    test(new Module {
      val io = IO(new Bundle {
        val mainOut = Output(UInt(8.W))

        val divClock = Output(Clock())
        val divOut = Output(UInt(8.W))
      })

      io.mainOut := Counter(true.B, 8)._1

      val divClock = RegInit(true.B)
      divClock := !divClock
      io.divClock := divClock.asClock

      withClock(io.divClock) {
        io.divOut := Counter(true.B, 8)._1
      }
    }) { c =>
      fork {
        c.io.mainOut.expect(0.U)
        c.io.divOut.expect(0.U)
        c.clock.step()
        c.io.mainOut.expect(1.U)
        c.io.divOut.expect(0.U)
        c.clock.step()
        c.io.mainOut.expect(2.U)
        c.io.divOut.expect(1.U)
        c.clock.step()
        c.io.mainOut.expect(3.U)
        c.io.divOut.expect(1.U)
      } .fork {
        c.io.mainOut.expect(0.U)
        c.io.divOut.expect(0.U)
        c.io.divClock.step()
        c.io.mainOut.expect(2.U)
        c.io.divOut.expect(1.U)
      } .join
    }
  }
}
