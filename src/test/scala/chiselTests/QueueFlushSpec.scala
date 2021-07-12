package chiselTests

import org.scalacheck._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
import chisel3.util.random.LFSR

class ThingsPassThroughFlushQueueTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
  
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))  
  q.io.flush.get := false.B
  val elems = VecInit(elements.map(_.U))

  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    //ensure that what comes out is what comes in
    assert(elems(outCnt.value) === q.io.deq.bits)
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueGetsFlushedTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
    val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
    val elems = VecInit(elements.map {
    _.asUInt()
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)
  val halfCnt = (queueDepth + 1)/2
  q.io.flush.get := (inCnt.value === halfCnt.U) || (inCnt.value === 0.U) || (inCnt.value === elements.length.U)
  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
  }

  when((inCnt.value === halfCnt.U) || (inCnt.value === 0.U) || (inCnt.value === elements.length.U)) {
    //check that queue gets flushed at the beginning of the list of elements, in the middle, and at the end
    assert(!q.io.deq.valid) 
    assert(q.io.enq.ready)  
  }
  
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueFlushSpec extends ChiselPropSpec {
  // Disable shrinking on error.
  implicit val noShrinkListVal = Shrink[List[Int]](_ => Stream.empty)
  implicit val noShrinkInt = Shrink[Int](_ => Stream.empty)
  
  property("Queue should have things pass through") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new ThingsPassThroughFlushQueueTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }

  property("Queue should flush when requested") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        runTester {
          new QueueGetsFlushedTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }
}
