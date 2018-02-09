package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.util._
import chisel3.testers2._
import chisel3.testers2.TestAdapters._

class QueueTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2 with Queue"

  it should "pass through elements" in {
    test(Firrterpreter.start(new Module {
      val io = IO(new Bundle {
        val in = Flipped(Decoupled(UInt(8.W)))
        val out = Decoupled(UInt(8.W))
      })
      io.out <> Queue(io.in)
    })) { c =>
      val source = new ReadyValidSource(c.io.in, c.clock)
      val sink = new ReadyValidSink(c.io.out, c.clock)

      source.enqueueNow(42.U)
      sink.checkInvalid()
      c.clock.step(1)
      sink.checkDequeueNow(42.U)
      c.clock.step(1)  // TODO: eliminate this once valid can be pulldown (low priority poke)
      sink.checkInvalid()
      source.enqueueNow(43.U)
      c.clock.step(1)
      sink.checkDequeueNow(43.U)
    }
  }
}
