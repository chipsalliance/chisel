// SPDX-License-Identifier: Apache-2.0

/**
 * ChiselTest Compatibility Layer for Chisel 7
 *
 * This package provides a drop-in replacement for the ChiselTest library that was removed in Chisel 7.
 * It preserves the familiar ChiselTest API while delegating to Chisel 7's ChiselSim underneath.
 *
 * Usage:
 * {{{
 * import chiseltest._
 * import org.scalatest.flatspec.AnyFlatSpec
 *
 * class MyTest extends AnyFlatSpec with ChiselScalatestTester {
 *   it should "work" in {
 *     test(new MyModule) { dut =>
 *       dut.io.in.poke(42.U)
 *       dut.clock.step()
 *       dut.io.out.expect(42.U)
 *     }
 *   }
 * }
 * }}}
 *
 * Key Components:
 * - testableData, testableUInt, testableBoolExt: Implicit conversions for poke/peek/expect
 * - testableClock: Clock stepping and control
 * - DecoupledIOOps: Utilities for Decoupled interface testing
 * - ChiselScalatestTester: ScalaTest integration trait
 *
 * See README.md for detailed documentation.
 */
package object chiseltest {
  import scala.language.implicitConversions
  import chisel3._
  import chisel3.simulator.EphemeralSimulator._
  import chisel3.util.DecoupledIO

  // Dummy annotations for compatibility (ignored in Chisel 7)
  case object WriteVcdAnnotation
  case object VerilatorBackendAnnotation

  // Re-export the implicit conversions from ChiselTestCompat
  implicit class testableData[T <: Data](val x: T) extends AnyVal {
    def poke(value: T): Unit =
      toTestableData(x).poke(value)

    def peek(): T = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).peek()
    }

    def expect(value: T): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(value)
    }

    def expect(value: T, message: String): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(value, message)
    }

    // Add pokePartial for Bundle Lit values
    def pokePartial(litBundle: T): Unit = {
      // When given a Lit bundle, poke it directly
      toTestableData(x).poke(litBundle)
    }

    // Add expectPartial for Bundle Lit values
    def expectPartial(litBundle: T): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(litBundle)
    }
  }

  // Enhanced UInt operations
  implicit class testableUInt(val x: UInt) extends AnyVal {
    def poke(value: UInt): Unit =
      toTestableData(x).poke(value)

    def poke(value: BigInt): Unit =
      toTestableData(x).poke(value.U)

    def poke(value: Int): Unit =
      toTestableData(x).poke(value.U)

    def peek(): UInt = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).peek().asInstanceOf[UInt]
    }

    def peekInt(): BigInt = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).peek().asInstanceOf[UInt].litValue
    }

    def expect(value: UInt): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(value)
    }

    def expect(value: BigInt): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(value.U)
    }

    def expect(value: Int): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(value.U)
    }
  }

  // Enhanced Bool operations
  implicit class testableBoolExt(val x: Bool) extends AnyVal {
    def poke(value: Bool): Unit =
      toTestableBool(x).poke(value)

    def poke(value: Boolean): Unit =
      toTestableBool(x).poke(value.B)

    def peek(): Bool = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableBool(x).peek()
    }

    def peekBoolean(): Boolean = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableBool(x).peek().litToBoolean
    }

    def expect(value: Bool): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableBool(x).expect(value)
    }

    def expect(value: Boolean): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableBool(x).expect(value.B)
    }
  }

  implicit class testableClock(val x: Clock) extends AnyVal {
    def step(): Unit =
      toTestableClock(x).step(1)

    def step(cycles: Int): Unit =
      toTestableClock(x).step(cycles)

    // Stub for setTimeout - ignored in Chisel 7
    def setTimeout(cycles: Int): Unit = {
      // ChiselSim doesn't have explicit timeouts, this is a no-op for compatibility
    }
  }

  implicit class testableReset(val x: Reset) extends AnyVal {
    def poke(value: Bool): Unit =
      toTestableReset(x).poke(value.asInstanceOf[Reset])
  }

  // Re-export DecoupledDriver implicits
  implicit class DecoupledIOOps[T <: Data](val x: DecoupledIO[T]) extends AnyVal {
    def enqueueNow(data: T)(implicit clock: Clock): Unit = {
      // Drive data and valid for one cycle
      // This allows enqueueing even if consumer is not ready (data goes into queue)
      toTestableData(x.bits).poke(data)
      toTestableBool(x.valid).poke(true.B)
      toTestableClock(clock).step(1)
      toTestableBool(x.valid).poke(false.B)
    }

    def enqueueSeq(data: Seq[T])(implicit clock: Clock): Unit =
      data.foreach { d =>
        enqueueNow(d)
      }

    def expectDequeueNow(data: T, maxCycles: Int = 1000)(implicit clock: Clock): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableBool(x.ready).poke(true.B)
      // Wait until valid is seen before checking data
      var waited = 0
      while (!toTestableBool(x.valid).peek().litToBoolean) {
        if (waited >= maxCycles) {
          throw new RuntimeException(s"expectDequeueNow: valid not asserted within $maxCycles cycles")
        }
        toTestableClock(clock).step(1)
        waited += 1
      }
      toTestableData(x.bits).expect(data)
      toTestableClock(clock).step(1)
      toTestableBool(x.ready).poke(false.B)
    }

    def expectDequeueSeq(data: Seq[T])(implicit clock: Clock): Unit =
      data.foreach { d =>
        expectDequeueNow(d)
      }

    def expectPeek(data: T): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x.bits).expect(data)
    }

    def expectInvalid(): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableBool(x.valid).expect(false.B)
    }

    def initSource()(implicit clock: Clock): Unit =
      toTestableBool(x.valid).poke(false.B)

    def initSink()(implicit clock: Clock): Unit =
      toTestableBool(x.ready).poke(false.B)
  }

  // Stub for fork - not fully supported but allows compilation
  def fork(body: => Unit): ForkHandle = {
    // Execute immediately in Chisel 7 (no true concurrency support yet)
    body
    new ForkHandle
  }

  class ForkHandle {
    def join(): Unit = ()
  }
}
