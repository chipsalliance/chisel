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
    val elems = VecInit(elements.map(_.U))

  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)
  
  val halfCnt = (queueDepth + 1)/2
  val cycleCounter = Counter(elements.length + 1)
  //testing a flush when 
  val flush = LFSR(16)((tap + 3) % 16)
  q.io.flush.get := flush
  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)
  cycleCounter.inc() //counts every cycle

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
  }

  when(flush) {
    //check that queue gets flushed
    assert((q.io.count === 0.U) || q.io.deq.valid, s"in count: ${q.io.count} and queue length: $queueDepth" ) 
    assert(!q.io.enq.ready, s"in count: ${q.io.count} and queue length: $queueDepth") 
  } 
  
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
    stop()
  }
}

class EmptyFlushEdgecaseTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
    val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
    val elems = VecInit(elements.map(_.U))

  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)
  
  val cycleCounter = Counter(elements.length + 1)
  //testing a flush when queue is empty
  val flush = (cycleCounter.value === 0.U && inCnt.value === 0.U) //flushed only before anything is enqueued  
  q.io.flush.get := flush
  cycleCounter.inc() //counts every cycle

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
  }

  when(flush) {
    //check that queue gets flushed at the beginning with no elements
    assert(!q.io.deq.valid, s"in count: ${q.io.count} and queue length: $queueDepth" ) 
    assert(!q.io.enq.ready, s"in count: ${q.io.count} and queue length: $queueDepth") 
  } 
  
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
    stop()
  }
}

class FullQueueFlushEdgecaseTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
    val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
    val elems = VecInit(elements.map(_.U))

  val inCnt = Counter(elements.length + 1)
  val currDepthCnt = Counter(queueDepth + 1)
  
  //testing a flush when queue is empty
  val flush = (currDepthCnt.value === queueDepth.U)
  q.io.flush.get := flush

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
    currDepthCnt.inc() //counts how many items have been enqueued
  }

  when(flush) {
    currDepthCnt.reset() //resets the number of items currently inside queue
    //check that queue gets flushed when queue is full
    assert(q.io.deq.valid, s"in count: ${q.io.count} and queue length: $queueDepth" ) 
    assert(!q.io.enq.ready, s"in count: ${q.io.count} and queue length: $queueDepth")
    
  } 
  
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
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
        assertTesterPasses {
          new QueueGetsFlushedTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }
  property("Queue flush when queue is empty") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new EmptyFlushEdgecaseTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }
  property("Queue flush when queue is full") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new FullQueueFlushEdgecaseTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }
}
