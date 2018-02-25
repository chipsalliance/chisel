package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.tester._

class ThreadJoinTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2 threading fork-joins"

  class PassthroughModule[T <: Data](ioType: T) extends Module {
    val io = IO(new Bundle {
      val in = Input(ioType)
      val out = Output(ioType)
    })
    io.out := io.in
  }

  it should "join a single thread" in {
    test(new PassthroughModule(UInt(8.W))) { c =>
      c.io.in.poke(15.U)
      fork {
        c.clock.step(1)
        c.io.in.poke(16.U)
        c.clock.step(1)  // needed to avoid thread collision errors
      } .join
      c.io.in.expect(16.U)
    }
  }

  it should "join multiple threads of uneven length, order 1" in {
    test(new PassthroughModule(UInt(8.W))) { c =>
      c.io.in.poke(15.U)
      fork {
        c.clock.step(3)
        c.io.in.poke(16.U)
        c.clock.step(1)  // needed to avoid thread collision errors
      } .fork {  // do-nothing thread finishes first
        c.clock.step(1)
      } .join
      c.io.in.expect(16.U)
    }
  }

  it should "join multiple threads of uneven length, order 2" in {
    test(new PassthroughModule(UInt(8.W))) { c =>
      c.io.in.poke(15.U)
      fork {  // do-nothing thread finishes first
        c.clock.step(1)
      } .fork {
        c.clock.step(3)
        c.io.in.poke(16.U)
        c.clock.step(1)  // needed to avoid thread collision errors
      } .join
      c.io.in.expect(16.U)
    }
  }
}
