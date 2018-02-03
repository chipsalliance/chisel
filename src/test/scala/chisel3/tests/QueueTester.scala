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
      c.io.in.enqueueNow(42.U, c.clock)
      c.io.out.expectDequeueNow(42.U, c.clock)
      c.io.in.enqueueNow(43.U, c.clock)
      c.io.out.expectDequeueNow(43.U, c.clock)
    }
  }
}
