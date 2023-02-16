// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
import chisel3.util.random.LFSR
import org.scalacheck._

class ThingsPassThroughTester(
  elements:       Seq[Int],
  queueDepth:     Int,
  bitWidth:       Int,
  tap:            Int,
  useSyncReadMem: Boolean,
  hasFlush:       Boolean)
    extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, useSyncReadMem = useSyncReadMem, hasFlush = hasFlush))
  val elems = VecInit(elements.map {
    _.asUInt
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)
  q.io.flush.foreach { _ := false.B } //Flush behavior is tested in QueueFlushSpec
  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire) {
    inCnt.inc()
  }
  when(q.io.deq.fire) {
    //ensure that what comes out is what comes in
    assert(elems(outCnt.value) === q.io.deq.bits)
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueReasonableReadyValid(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean)
    extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, useSyncReadMem = useSyncReadMem))
  val elems = VecInit(elements.map {
    _.asUInt
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
  when(q.io.enq.fire) {
    inCnt.inc()
  }
  when(q.io.deq.fire) {
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class CountIsCorrectTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean)
    extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, useSyncReadMem = useSyncReadMem))
  val elems = VecInit(elements.map {
    _.asUInt(bitWidth.W)
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire) {
    inCnt.inc()
    assert(q.io.count === (inCnt.value - outCnt.value))
  }
  when(q.io.deq.fire) {
    outCnt.inc()
    assert(q.io.count === (inCnt.value - outCnt.value))
  }
  //assert(q.io.count === (inCnt.value - outCnt.value))

  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueSinglePipeTester(elements: Seq[Int], bitWidth: Int, tap: Int, useSyncReadMem: Boolean) extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), 1, pipe = true, useSyncReadMem = useSyncReadMem))
  val elems = VecInit(elements.map {
    _.asUInt(bitWidth.W)
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  assert(q.io.enq.ready || (q.io.count === 1.U && !q.io.deq.ready))

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire) {
    inCnt.inc()
  }
  when(q.io.deq.fire) {
    outCnt.inc()
  }

  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueuePipeTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean)
    extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, pipe = true, useSyncReadMem = useSyncReadMem))
  val elems = VecInit(elements.map {
    _.asUInt(bitWidth.W)
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  q.io.deq.ready := LFSR(16)(tap)

  assert(q.io.enq.ready || (q.io.count === queueDepth.U && !q.io.deq.ready))

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire) {
    inCnt.inc()
  }
  when(q.io.deq.fire) {
    outCnt.inc()
  }

  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueFlowTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean)
    extends BasicTester {
  val q = Module(new Queue(UInt(bitWidth.W), queueDepth, flow = true, useSyncReadMem = useSyncReadMem))
  val elems = VecInit(elements.map {
    _.asUInt
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  q.io.enq.valid := (inCnt.value < elements.length.U)
  //Queue should be full or ready
  assert(q.io.enq.ready || q.io.count === queueDepth.U)

  q.io.deq.ready := LFSR(16)(tap)
  //Queue should be empty or valid
  assert(q.io.deq.valid || (q.io.count === 0.U && !q.io.enq.fire))

  q.io.enq.bits := elems(inCnt.value)
  when(q.io.enq.fire) {
    inCnt.inc()
  }
  when(q.io.deq.fire) {
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueFactoryTester(elements: Seq[Int], queueDepth: Int, bitWidth: Int, tap: Int, useSyncReadMem: Boolean)
    extends BasicTester {
  val enq = Wire(Decoupled(UInt(bitWidth.W)))
  val deq = Queue(enq, queueDepth, useSyncReadMem = useSyncReadMem)

  val elems = VecInit(elements.map {
    _.asUInt
  })
  val inCnt = Counter(elements.length + 1)
  val outCnt = Counter(elements.length + 1)

  enq.valid := (inCnt.value < elements.length.U)
  deq.ready := LFSR(16)(tap)

  enq.bits := elems(inCnt.value)
  when(enq.fire) {
    inCnt.inc()
  }
  when(deq.fire) {
    //ensure that what comes out is what comes in
    assert(elems(outCnt.value) === deq.bits)
    outCnt.inc()
  }
  when(outCnt.value === elements.length.U) {
    stop()
  }
}

class QueueSpec extends ChiselPropSpec {

  property("Queue should have things pass through") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new ThingsPassThroughTester(se._2, depth, se._1, tap, isSync, false)
        }
      }
    }
  }

  property("Queue should have reasonable ready/valid") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueueReasonableReadyValid(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }

  property("Queue should have correct count") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new CountIsCorrectTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }

  property("Queue pipe should work for 1-element queues") {
    forAll(safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (se, tap, isSync) =>
      whenever(se._1 >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueueSinglePipeTester(se._2, se._1, tap, isSync)
        }
      }
    }
  }

  property("Queue pipe should work for more general queues") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueuePipeTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }

  property("Queue flow should work") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && depth >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueueFlowTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }

  property("Queue companion object factory method should work") {
    forAll(vecSizes, safeUIntN(20), Gen.choose(0, 15), Gen.oneOf(true, false)) { (depth, se, tap, isSync) =>
      whenever(se._1 >= 1 && se._2.nonEmpty) {
        assertTesterPasses {
          new QueueFactoryTester(se._2, depth, se._1, tap, isSync)
        }
      }
    }
  }

  property("Queue.irrevocable should elaborate") {
    class IrrevocableQueue extends Module {
      val in = Wire(Decoupled(Bool()))
      val iQueue = Queue.irrevocable(in, 1)
    }
    (new chisel3.stage.phases.Elaborate)
      .transform(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new IrrevocableQueue)))
  }

  property("Queue.apply should have decent names") {
    class HasTwoQueues extends Module {
      val in = IO(Flipped(Decoupled(UInt(8.W))))
      val out = IO(Decoupled(UInt(8.W)))

      val foo = Queue(in, 2)
      val bar = Queue(foo, 2)
      out <> bar
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new HasTwoQueues)
    chirrtl should include("inst foo_q of Queue")
    chirrtl should include("inst bar_q of Queue")
  }
}
