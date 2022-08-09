// See README.md for license details.

package chisel3.std.fifo

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.formal._
import firrtl.AnnotationSeq
import org.scalatest.flatspec.AnyFlatSpec

class FifoFormalTest extends AnyFlatSpec with ChiselScalatestTester with Formal {
  private val defaultOptions: AnnotationSeq = Seq(BoundedCheck(10), BtormcEngineAnnotation)

  "BubbleFifo" should "pass" in {
    verify(new FifoTestWrapper(new BubbleFifo(UInt(16.W), 4)), defaultOptions)
  }

  "DoubleBufferFifo" should "pass" in {
    verify(new FifoTestWrapper(new DoubleBufferFifo(UInt(16.W), 4)), defaultOptions)
  }

  "RegFifo" should "pass" in {
    verify(new FifoTestWrapper(new RegFifo(UInt(16.W), 4)), defaultOptions)
  }

  "MemFifo" should "pass" in {
    verify(new FifoTestWrapper(new MemFifo(UInt(16.W), 4)), defaultOptions)
  }

  "CombFifo" should "pass" in {
    verify(new FifoTestWrapper(new CombFifo(UInt(16.W), 4)), defaultOptions)
  }
}

class FifoTestWrapper(fifo: => Fifo[UInt]) extends Module {
  val dut = Module(fifo)
  val tracker = Module(new MagicPacketTracker(dut.depth))
  val io = IO(chiselTypeOf(dut.io))
  io <> dut.io
  tracker.io := io
  val startTracking = IO(Input(Bool()))
  tracker.startTracking := startTracking
}

/** Tracks random packets for formally verifying FIFOs
  *
  *  This ensures that when some data enters the FIFO, it
  *  will always be dequeued after the correct number of
  *  elements.
  *  So essentially we are verifying data integrity.
  *  Note that this does not imply that the FIFO has no bugs
  *  since e.g., a FIFO that never allows elements to be
  *  enqueued would easily pass our assertions.
  */
class MagicPacketTracker(fifoDepth: Int) extends Module {
  val io = IO(Input(new FifoIO[UInt](UInt())))

  // count the number of elements in the fifo
  val elementCount = RegInit(0.U(log2Ceil(fifoDepth + 1).W))
  val nextElementCount = Mux(
    io.enq.fire && !io.deq.fire,
    elementCount + 1.U,
    Mux(!io.enq.fire && io.deq.fire, elementCount - 1.U, elementCount)
  )
  elementCount := nextElementCount

  // track a random "magic" packet through the fifo
  val startTracking = IO(Input(Bool()))
  val isActive = RegInit(false.B)
  val packetValue = Reg(chiselTypeOf(io.enq.bits))
  val packetCount = Reg(chiselTypeOf(elementCount))

  when(!isActive && io.enq.fire && startTracking) {
    when(io.deq.fire && elementCount === 0.U) {
      assert(io.enq.bits === io.deq.bits, "element should pass through the fifo")
    }.otherwise {
      isActive := true.B
      packetValue := io.enq.bits
      packetCount := nextElementCount
    }
  }

  when(isActive && io.deq.fire) {
    packetCount := packetCount - 1.U
    when(packetCount === 1.U) {
      assert(packetValue === io.deq.bits, "element should be dequeued in this cycle")
      isActive := false.B
    }
  }
}
