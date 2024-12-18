package chiselTests

import org.scalacheck._

import chisel3._
import chisel3.testers.{BasicTester, TesterDriver}
import chisel3.util._
import chisel3.util.random.LFSR

/** Test elements can be enqueued and dequeued when flush is tied to false
  *
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
class ThingsPassThroughFlushQueueTester(
  elements:       Seq[Int],
  queueDepth:     Int,
  bitWidth:       Int,
  tap:            Int,
  useSyncReadMem: Boolean)
    extends ThingsPassThroughTester(elements, queueDepth, bitWidth, tap, useSyncReadMem, hasFlush = true)

/** Generic flush queue tester base class
  *
  * @param elements The sequence of elements used in the queue
  * @param queueDepth The max number of entries in the queue
  * @param bitWidth Integer size of the data type used in the queue
  * @param tap Integer tap('seed') for the LFSR
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element
  */
abstract class FlushQueueTesterBase(
  elements:       Seq[Int],
  queueDepth:     Int,
  bitWidth:       Int,
  tap:            Int,
  useSyncReadMem: Boolean)
    extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, hasFlush = true))
  val elems = VecInit(elements.map(_.U))
  val inCnt = Counter(elements.length + 1)
  val outCnt = RegInit(0.U(log2Ceil(elements.length).W))
  val currQCnt = RegInit(0.U(log2Ceil(5).W))

  val flush: Bool = WireInit(false.B)
  val flushRegister = RegNext(flush, init = false.B)
  q.io.flush.get := flush
  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire) {
    inCnt.inc()
    currQCnt := currQCnt + 1.U //counts how many items have been enqueued
  }
  when(q.io.deq.fire) {
    assert(flushRegister === false.B) //check queue isn't flushed (can't dequeue an empty queue)
  }
  when(flushRegister) { //Internal signal maybe_full is a register so some signals update on the next cycle
    //check that queue gets flushed when queue is full
    assert(q.io.count === 0.U)
    assert(!q.io.deq.valid, "Expected to not be able to dequeue when flush is asserted the previous cycle")
    assert(
      q.io.enq.ready,
      "Expected enqueue to be ready when flush was asserted the previous cycle because queue should be empty"
    )
  }
  when(inCnt.value === elements.length.U) { //stop when all entries are enqueued
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
class QueueGetsFlushedTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean)
    extends FlushQueueTesterBase(elements, queueDepth, bitWidth, tap, useSyncReadMem) {
  flush := LFSR(16)((tap + 3) % 16) //testing a flush when flush is called randomly
  val halfCnt = (queueDepth + 1) / 2

  when(q.io.deq.fire) {
    //ensure that what comes out is what comes in
    assert(currQCnt <= queueDepth.U)
    assert(elems(outCnt) === q.io.deq.bits)
    outCnt := outCnt + 1.U
    when(currQCnt > 0.U) {
      currQCnt := Mux(q.io.enq.fire, currQCnt, (currQCnt - 1.U))
    }
  }
  when(flush) {
    assert(currQCnt === 0.U || q.io.deq.valid)
    outCnt := outCnt + Mux(q.io.enq.fire, (currQCnt + 1.U), currQCnt)
    currQCnt := 0.U //resets the number of items currently inside queue
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
class EmptyFlushEdgecaseTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean)
    extends FlushQueueTesterBase(elements, queueDepth, bitWidth, tap, useSyncReadMem) {
  val cycleCounter = Counter(elements.length + 1)
  cycleCounter.inc() //counts every cycle

  //testing a flush when queue is empty
  flush := (cycleCounter.value === 0.U && inCnt.value === 0.U) //flushed only before anything is enqueued
  q.io.enq.valid := (inCnt.value < elements.length.U) && !flush

  when(q.io.deq.fire) {
    assert(elems(outCnt) === q.io.deq.bits)
    outCnt := outCnt + 1.U
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
class EnqueueEmptyFlushEdgecaseTester(
  elements:       Seq[Int],
  queueDepth:     Int,
  bitWidth:       Int,
  tap:            Int,
  useSyncReadMem: Boolean)
    extends FlushQueueTesterBase(elements, queueDepth, bitWidth, tap, useSyncReadMem) {
  val cycleCounter = Counter(elements.length + 1)
  val outCounter = Counter(elements.length + 1)

  //testing an enqueue during a flush
  flush := (cycleCounter.value === 0.U && inCnt.value === 0.U) //flushed only before anything is enqueued
  cycleCounter.inc() //counts every cycle

  when(q.io.deq.fire) {
    //flush and enqueue were both active on the first cycle,
    //so that element is flushed immediately which makes outCnt off by one
    assert(elems(outCounter.value + 1.U) === q.io.deq.bits) //ensure that what comes out is what comes in
    outCounter.inc()
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
class FullQueueFlushEdgecaseTester(
  elements:       Seq[Int],
  queueDepth:     Int,
  bitWidth:       Int,
  tap:            Int,
  useSyncReadMem: Boolean)
    extends FlushQueueTesterBase(elements, queueDepth, bitWidth, tap, useSyncReadMem) {

  //testing a flush when queue is full
  flush := (currQCnt === queueDepth.U)

  when(q.io.deq.fire) {
    //ensure that what comes out is what comes in
    assert(currQCnt <= queueDepth.U)
    assert(elems(outCnt) === q.io.deq.bits)
    outCnt := outCnt + 1.U
    when(currQCnt > 0.U) {
      currQCnt := currQCnt - 1.U
    }
  }
  when(flush) {
    outCnt := outCnt + currQCnt
    currQCnt := 0.U //resets the number of items currently inside queue
    assert(currQCnt === 0.U || q.io.deq.valid)
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
class DequeueFullQueueEdgecaseTester(
  elements:       Seq[Int],
  queueDepth:     Int,
  bitWidth:       Int,
  tap:            Int,
  useSyncReadMem: Boolean)
    extends FlushQueueTesterBase(elements, queueDepth, bitWidth, tap, useSyncReadMem) {
  //Queue should be able to dequeue when queue is not empty and flush is high

  //testing a flush when dequeue is called
  flush := currQCnt === (queueDepth / 2).U
  q.io.enq.valid := !flushRegister
  q.io.deq.ready := flush

  when(q.io.deq.fire) {
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
    assert(q.io.deq.fire === false.B)
  }

}

class QueueFlushSpec extends ChiselPropSpec {

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
        assertTesterPasses(
          new DequeueFullQueueEdgecaseTester(se._2, depth, se._1, tap, isSync)
        )
      }
    }
  }
}
