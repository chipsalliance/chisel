// SPDX-License-Identifier: Apache-2.0

package chiselTests.chiseltest

import chisel3._
import chiseltest._
import chiseltest.examples._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * Basic tests for the chiseltest examples
 */
class ExamplesBasicTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "PassthroughModule"

  it should "pass through input to output" in {
    test(new PassthroughModule) { dut =>
      dut.io.in.poke(42.U)
      dut.io.out.expect(42.U)
    }
  }

  it should "handle 0 input" in {
    test(new PassthroughModule) { dut =>
      dut.io.in.poke(0.U)
      dut.io.out.expect(0.U)
    }
  }

  it should "handle max 8-bit value" in {
    test(new PassthroughModule) { dut =>
      dut.io.in.poke(255.U)
      dut.io.out.expect(255.U)
    }
  }
}

/**
 * Tests for the SimpleCounter module
 */
class ExamplesCounterTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SimpleCounter"

  it should "count when enabled" in {
    test(new SimpleCounter) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      dut.io.en.poke(true.B)
      dut.io.out.expect(0.U)

      for (i <- 1 to 5) {
        dut.clock.step(1)
        dut.io.out.expect(i.U)
      }
    }
  }

  it should "hold value when disabled" in {
    test(new SimpleCounter) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      dut.io.en.poke(true.B)
      dut.clock.step(3)
      dut.io.en.poke(false.B)
      dut.clock.step(1)
      dut.io.out.expect(3.U)

      dut.clock.step(2)
      dut.io.out.expect(3.U)
    }
  }
}

/**
 * Tests for the SimpleRegister module
 */
class ExamplesRegisterTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SimpleRegister"

  it should "store and retrieve data" in {
    test(new SimpleRegister) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      dut.io.data_in.poke(42.U)
      dut.io.write_en.poke(true.B)
      dut.clock.step(1)
      dut.io.data_out.expect(42.U)
    }
  }

  it should "not write when disabled" in {
    test(new SimpleRegister) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      dut.io.data_in.poke(42.U)
      dut.io.write_en.poke(true.B)
      dut.clock.step(1)
      dut.io.data_in.poke(100.U)
      dut.io.write_en.poke(false.B)
      dut.clock.step(1)
      dut.io.data_out.expect(42.U)
    }
  }
}

/**
 * Tests for the SimpleQueue module
 */
class ExamplesQueueTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SimpleQueue"

  it should "enqueue and dequeue a value" in {
    test(new SimpleQueue) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      // Initially not valid
      dut.io.deq_valid.expect(false.B)
      dut.io.enq_ready.expect(true.B)

      // Enqueue a value
      dut.io.enq_data.poke(42.U)
      dut.io.enq_valid.poke(true.B)
      dut.clock.step(1)
      dut.io.enq_valid.poke(false.B)

      // Check dequeue valid and data
      dut.io.deq_valid.expect(true.B)
      dut.io.deq_data.expect(42.U)

      // Dequeue
      dut.io.deq_ready.poke(true.B)
      dut.clock.step(1)

      // Should be empty again
      dut.io.deq_valid.expect(false.B)
    }
  }

  it should "hold data until dequeued" in {
    test(new SimpleQueue) { dut =>
      dut.reset.poke(true.B)
      dut.clock.step(1)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      // Enqueue
      dut.io.enq_data.poke(99.U)
      dut.io.enq_valid.poke(true.B)
      dut.clock.step(1)

      // Don't dequeue, just step
      dut.io.enq_valid.poke(false.B)
      dut.io.deq_ready.poke(false.B)
      dut.clock.step(2)

      // Data should still be there
      dut.io.deq_valid.expect(true.B)
      dut.io.deq_data.expect(99.U)
    }
  }
}
