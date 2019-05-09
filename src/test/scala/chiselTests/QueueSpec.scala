// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._
import org.scalacheck._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
import chisel3.util.random.LFSR

class ThingsPassThroughTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth))
  val elems = VecInit(elements.map {
    _.asUInt()
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    //ensure that what comes otu is what comes in
    assert(elems(outCnt.value) === q.io.deq.bits)
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueReasonableReadyValid(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth))
  val elems = VecInit(elements.map {
    _.asUInt()
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  //Queue should be full or ready
  assert(q.io.enq.ready || q.io.count === queueDepth.U)

  q.io.deq.ready := LFSR(16)(tap)
  //Queue should be empty or valid
  assert(q.io.deq.valid || q.io.count === 0.U)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class CountIsCorrectTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth))
  val elems = VecInit(elements.map {
    _.asUInt(bitWidth.W)
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
    assert(q.io.count === (inCnt.value - outCnt.value))
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
    assert(q.io.count === (inCnt.value - outCnt.value))
  }
  //assert(q.io.count === (inCnt.value - outCnt.value))

  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueSinglePipeTester(elements: Seq[Int], bitWidth: Int, tap: Int) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), 1, pipe = true))
  val elems = VecInit(elements.map {
    _.asUInt(bitWidth.W)
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  assert(q.io.enq.ready || (q.io.count === 1.U && !q.io.deq.ready))

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
  }

  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueuePipeTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, pipe = true))
  val elems = VecInit(elements.map {
    _.asUInt(bitWidth.W)
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  assert(q.io.enq.ready || (q.io.count === queueDepth.U && !q.io.deq.ready))

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
  }

  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueFlowTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, flow = true))
  val elems = VecInit(elements.map {
    _.asUInt()
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  //Queue should be full or ready
  assert(q.io.enq.ready || q.io.count === queueDepth.U)

  q.io.deq.ready := LFSR(16)(tap)
  //Queue should be empty or valid
  assert(q.io.deq.valid || (q.io.count === 0.U && !q.io.enq.fire()))

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire()) {
    inCnt.inc()
  }
  when(q.io.deq.fire()) {
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueSpec extends ChiselPropSpec {
  // Disable shrinking on error.
  implicit val noShrinkListVal = Shrink[List[Int]](_ => Stream.empty)
  implicit val noShrinkInt = Shrink[Int](_ => Stream.empty)

  property("Queue should have things pass through") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15)) { (depth, se, tap) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new ThingsPassThroughTester(se._2, depth, se._1, tap)
        }
      }
    }
  }

  property("Queue should have reasonable ready/valid") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15)) { (depth, se, tap) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueueReasonableReadyValid(se._2, depth, se._1, tap)
        }
      }
    }
  }

  property("Queue should have correct count") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15)) { (depth, se, tap) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new CountIsCorrectTester(se._2, depth, se._1, tap)
        }
      }
    }
  }

  property("Queue pipe should work for 1-element queues") {
    forAll(safeUIntN(20), Gen.choose(0, 15)) { (se, tap) =>
      whenever(se._1 >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueueSinglePipeTester(se._2, se._1, tap)
        }
      }
    }
  }

  property("Queue pipe should work for more general queues") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15)) { (depth, se, tap) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueuePipeTester(se._2, depth, se._1, tap)
        }
      }
    }
  }

  property("Queue flow should work") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15)) { (depth, se, tap) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueueFlowTester(se._2, depth, se._1, tap)
        }
      }
    }
  }
}
