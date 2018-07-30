package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.tester._

class TimescopeTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2 timescopes"

  it should "revert signals at the end" in {
    test(new Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
    }) { c =>
      c.io.in.poke(0.U)
      c.io.out.expect(0.U)
      timescope {
        c.io.in.poke(1.U)
        c.io.out.expect(1.U)
        c.clock.step()
        c.io.out.expect(1.U)
      }
      c.io.out.expect(0.U)
      c.clock.step()
      c.io.out.expect(0.U)
    }
  }

  it should "allow combinational operations within" in {
    test(new Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
    }) { c =>
      c.io.in.poke(0.U)
      c.io.out.expect(0.U)
      timescope {
        c.io.in.poke(1.U)
        c.io.out.expect(1.U)
        c.io.in.poke(2.U)
        c.io.out.expect(2.U)
        c.clock.step()
        c.io.out.expect(2.U)
      }
      c.io.out.expect(0.U)
      c.io.in.poke(3.U)
      c.io.out.expect(3.U)
      c.clock.step()
      c.io.out.expect(3.U)
    }
  }
}
