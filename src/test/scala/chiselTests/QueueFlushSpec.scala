package chiselTests

import org.scalacheck._

import chisel3._
import chisel3.testers.{BasicTester, TesterDriver}
import chisel3.util._
import chisel3.util.random.LFSR
import treadle.WriteVcdAnnotation

/** Test elements can be enqueued and dequeued
 * 
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
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

/** Test queue can flush at random times
 * 
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
class QueueGetsFlushedTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
  
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
  val elems = VecInit(elements.map(_.U))
  val inCnt = Counter(elements.length + 1)
  val outCnt = RegInit(0.U(log2Ceil(elements.length).W))
  val currQCnt = RegInit(0.U(log2Ceil(5).W))
  val halfCnt = (queueDepth + 1)/2

  val flush = LFSR(16)((tap + 3) % 16)  //testing a flush when flush is called randomly
  q.io.flush.get := flush
  val flushRegister = RegInit(false.B)
  flushRegister := flush
  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
    currQCnt := currQCnt + 1.U //counts how many items have been enqueued
  }
  when(q.io.deq.fire()) {
    //ensure that what comes out is what comes in
    assert(flushRegister === false.B)
    assert(currQCnt <= queueDepth.U)
    assert(elems(outCnt) === q.io.deq.bits)
    outCnt := outCnt + 1.U
    when (currQCnt > 0.U) {
      currQCnt := Mux(q.io.enq.fire(), currQCnt, (currQCnt - 1.U))
    }
  }
  when(flush) {
    outCnt := outCnt + Mux(q.io.enq.fire(), (currQCnt + 1.U), currQCnt)
    currQCnt := 0.U //resets the number of items currently inside queue
    assert(currQCnt === 0.U || q.io.deq.valid)
  }
  when(flushRegister) { //Internal signal maybe_full is a register so some signals update on the next cycle
    //check that queue gets flushed when queue is full
    assert(q.io.count === 0.U)
    assert(!q.io.deq.valid, "Expected to not be able to dequeue when flush is asserted the previous cycle")
    assert(q.io.enq.ready, "Expected enqueue to be ready when flush was asserted the previous cycle because queue should be empty")
  } 
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
    stop()
  }
}

/** Test queue can flush when empty
 * 
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
class EmptyFlushEdgecaseTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
    val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
    val elems = VecInit(elements.map(_.U))

  val inCnt = Counter(elements.length + 1)
  val outCnt = RegInit(0.U(log2Ceil(elements.length).W))
  val cycleCounter = Counter(elements.length + 1)

  //testing a flush when queue is empty
  val flush = (cycleCounter.value === 0.U && inCnt.value === 0.U) //flushed only before anything is enqueued  
  q.io.flush.get := flush
  val flushRegister = RegInit(false.B)
  flushRegister := flush
  cycleCounter.inc() //counts every cycle

  q.io.enq.valid := (inCnt.value < elements.length.U) && !flush
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    assert(elems(outCnt) === q.io.deq.bits)
    outCnt := outCnt + 1.U
  }
  when(flushRegister) {
    //check that queue gets flushed at the beginning with no elements
    assert(!q.io.deq.valid, "Expected to not be able to dequeue when flush is asserted the previous cycle")
    assert(q.io.enq.ready, "Expected enqueue to be ready when flush was asserted the previous cycle because queue should be empty")
    assert(q.io.count === 0.U)
  } 
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
    stop()
  }
}

/** Test queue can enqueue during a flush
 * 
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
class EnqueueEmptyFlushEdgecaseTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
  val elems = VecInit(elements.map(_.U))
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)
  val cycleCounter = Counter(elements.length + 1)

  //testing an enqueue during a flush
  val flush = (cycleCounter.value === 0.U && inCnt.value === 0.U) //flushed only before anything is enqueued  
  q.io.flush.get := flush
  val flushRegister = RegInit(false.B)
  flushRegister := flush
  cycleCounter.inc() //counts every cycle

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    //flush and enqueue were both active on the first cycle, 
    //so that element is flushed immediately which makes outCnt off by one
    assert(elems(outCnt.value + 1.U) === q.io.deq.bits) //ensure that what comes out is what comes in
    outCnt.inc()
  }
  when(flushRegister) {
    //check that queue gets flushed at the beginning with no elements
    assert(!q.io.deq.valid, "Expected to not be able to dequeue when flush is asserted the previous cycle") 
    assert(q.io.enq.ready, "Expected enqueue to be ready when flush was asserted the previous cycle because queue should be empty")
    assert(q.io.count === 0.U)
  } 
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
    stop()
  }
}

/** Test queue can flush when full
 * 
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
class FullQueueFlushEdgecaseTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
  val elems = VecInit(elements.map(_.U))
  val inCnt = Counter(elements.length + 1)
  val outCnt = RegInit(0.U(log2Ceil(elements.length).W))
  val currQCnt = RegInit(0.U(log2Ceil(5).W))

  //testing a flush when queue is full
  val flush = (currQCnt === queueDepth.U)
  val flushRegister = RegInit(false.B)
  q.io.flush.get := flush
  flushRegister := flush
  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)
  q.io.enq.bits := elems(inCnt.value)

  when(q.io.enq.fire()) {
    inCnt.inc()
    currQCnt := currQCnt + 1.U //counts how many items have been enqueued
  }
  when(q.io.deq.fire()) {
    //ensure that what comes out is what comes in
    assert(currQCnt <= queueDepth.U)
    assert(elems(outCnt) === q.io.deq.bits)
    outCnt := outCnt + 1.U
    when (currQCnt > 0.U) {
      currQCnt := currQCnt - 1.U
    }
  }
  when(flush) {
    outCnt := outCnt + currQCnt
    currQCnt := 0.U //resets the number of items currently inside queue
    assert(currQCnt === 0.U || q.io.deq.valid)
  }
  when(flushRegister) { //Internal signal maybe_full is a register so some signals update on the next cycle
    //check that queue gets flushed when queue is full
    assert(q.io.count === 0.U)
    assert(!q.io.deq.valid, "Expected to not be able to dequeue when flush is asserted the previous cycle")
    assert(q.io.enq.ready, "Expected enqueue to be ready when flush is high because queue should be empty")
  }
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
    stop()
  }
}

/** Test queue can dequeue on the same cycle as a flush
 * 
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
class DequeueFullQueueEdgecaseTester (elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
  //Queue should be able to dequeue when queue is not empty and flush is high

  val elems = VecInit(elements.map(_.U))
  val inCnt = Counter(elements.length + 1)
  val outCnt = RegInit(0.U(log2Ceil(elements.length).W))
  val currQCnt = RegInit(0.U(log2Ceil(5).W))
  //testing a flush when dequeue is called
  val flush = currQCnt === (queueDepth/2).U
  val flushRegister = RegInit(false.B)
  q.io.flush.get := flush
  flushRegister := flush
  q.io.enq.valid := !flushRegister
  q.io.deq.ready := flush

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
    currQCnt := currQCnt + 1.U //counts how many items have been enqueued
  }
  when(q.io.deq.fire()) {
    //ensure that what comes out is what comes in
    assert(currQCnt <= queueDepth.U)
    assert(elems(outCnt) === q.io.deq.bits)
    assert(currQCnt > 0.U)
  }
  when(flush) {
    //The outcount register is one count behind because the dequeue happens at the same time as the flush
    outCnt := outCnt + currQCnt + 1.U 
    currQCnt := 0.U //resets the number of items currently inside queue
    assert(currQCnt === 0.U || q.io.deq.valid)
  }
  when(flushRegister) {
    //check that queue gets flushed when queue is full
    assert(q.io.deq.fire() === false.B)
    assert(q.io.count === 0.U)
    assert(!q.io.deq.valid, "Expected to not be able to dequeue a flushed queue after flush is set to high the previous cycle")
    assert(q.io.enq.ready, "Expected enqueue to be ready when flush was asserted the previous cycle because queue should be empty")
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
  property("Test queue can enqueue during a flush") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new EnqueueEmptyFlushEdgecaseTester(se._2, depth, se._1, tap, isSync)
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
  property("Queue should be able to dequeue when flush is high") {
    forAll(Gen.choose(3, 5), safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses (
          new DequeueFullQueueEdgecaseTester(se._2, depth, se._1, tap, isSync),
          annotations = Seq(WriteVcdAnnotation)
        )
      }
    }
  }
}
