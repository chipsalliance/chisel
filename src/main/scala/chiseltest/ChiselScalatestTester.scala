// SPDX-License-Identifier: Apache-2.0

package chiseltest

import chisel3._
// Keep EphemeralSimulator imports for implicit toTestable* conversions
import chisel3.simulator.EphemeralSimulator._
import scala.language.implicitConversions

/**
 * ChiselTest-compatible API that delegates to ChiselSim (Chisel 7)
 *
 * This trait provides the ChiselTest API that users are familiar with from Chisel 6,
 * but internally uses ChiselSim from Chisel 7 to perform the actual testing.
 *
 * AUTOMATIC RESET FEATURE:
 * This trait automatically resets the module before running tests,
 * mimicking ChiselTest's behavior from Chisel 6. This helps prevent issues with
 * uninitialized registers that can cause tests to fail.
 *
 * To enable/disable auto-reset or customize the reset duration:
 * {{{
 * class MyTest extends AnyFlatSpec with ChiselScalatestTester {
 *   override def autoResetEnabled: Boolean = false  // Disable auto-reset
 *   override def resetCycles: Int = 5               // Or change duration
 * }
 * }}}
 *
 * Example usage:
 * {{{
 * import chiseltest._
 * import org.scalatest.flatspec.AnyFlatSpec
 *
 * class MyModuleSpec extends AnyFlatSpec with ChiselScalatestTester {
 *   behavior of "MyModule"
 *
 *   it should "work correctly" in {
 *     test(new MyModule) { dut =>
 *       // Module is already reset at this point
 *       dut.io.in.poke(42.U)
 *       dut.clock.step()
 *       dut.io.out.expect(42.U)
 *     }
 *   }
 * }
 * }}}
 */
trait ChiselScalatestTester {

  /**
   * Enable automatic reset before each test.
   * Override this to disable auto-reset if needed.
   */
  def autoResetEnabled: Boolean = false

  /**
   * Number of clock cycles to assert reset.
   * Override this to change reset duration.
   */
  def resetCycles: Int = 1

  /**
   * Test a module with the given stimulus
   *
   * This method provides the ChiselTest interface.
   *
   * @param dutGen A generator function that creates the device under test
   * @tparam T The type of module being tested
   */
  def test[T <: Module](dutGen: => T): TestBuilder[T] =
    new TestBuilder(dutGen, autoResetEnabled, resetCycles)

  /**
   * Builder class to support .withAnnotations() chaining
   */
  class TestBuilder[T <: Module](dutGen: => T, autoReset: Boolean, resetCyc: Int) {
    def withAnnotations(annotations: Seq[Any]): TestRunner[T] = {
      // Annotations are ignored in Chisel 7 (for compatibility only)
      new TestRunner(dutGen, autoReset, resetCyc)
    }

    // Allow direct execution without annotations
    def apply(body: T => Unit): Unit = {
      simulate(dutGen) { dut =>
        if (autoReset) {
          applyReset(dut, resetCyc)
        }
        body(dut)
      }
    }
  }

  /**
   * Runner class that executes the test
   */
  class TestRunner[T <: Module](dutGen: => T, autoReset: Boolean, resetCyc: Int) {
    def apply(body: T => Unit): Unit = {
      simulate(dutGen) { dut =>
        if (autoReset) {
          applyReset(dut, resetCyc)
        }
        body(dut)
      }
    }
  }

  /**
   * Apply reset sequence to the DUT
   */
  private def applyReset[T <: Module](dut: T, cycles: Int): Unit = {
    // Use ChiselSim testable helpers directly to avoid implicit ambiguity
    toTestableReset(dut.reset).poke(true.B)
    toTestableClock(dut.clock).step(cycles)
    toTestableReset(dut.reset).poke(false.B)
  }
}

/** Enhanced Data operations compatible with ChiselTest API */
object ChiselTestCompat {
  import chisel3.simulator.EphemeralSimulator._

  /**
   * Implicit class to add ChiselTest-style operations to Data types
   *
   * This provides the `poke`, `peek`, `expect` methods that ChiselTest users
   * are familiar with.
   */
  implicit class testableData[T <: Data](val x: T) extends AnyVal {

    /**
     * Poke a value onto a port
     *
     * @param value The value to poke (as a Chisel literal)
     */
    def poke(value: T): Unit =
      toTestableData(x).poke(value)

    /**
     * Peek the current value of a port
     *
     * @return The current value as a Chisel literal
     */
    def peek(): T = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).peek()
    }

    /**
     * Expect a specific value on a port
     *
     * @param value The expected value
     */
    def expect(value: T): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(value)
    }

    /**
     * Expect a specific value on a port with a custom message
     *
     * @param value The expected value
     * @param message Custom error message if expectation fails
     */
    def expect(value: T, message: String): Unit = {
      implicit val si: chisel3.experimental.SourceInfo = chisel3.experimental.SourceInfo.materialize
      toTestableData(x).expect(value, message)
    }
  }

  /** Implicit class to add clock stepping operations */
  implicit class testableClock(val x: Clock) extends AnyVal {

    /** Step the clock by one cycle */
    def step(): Unit =
      toTestableClock(x).step(1)

    /**
     * Step the clock by a specified number of cycles
     *
     * @param cycles Number of cycles to step
     */
    def step(cycles: Int): Unit =
      toTestableClock(x).step(cycles)
  }

  /** Implicit class to add reset operations */
  implicit class testableReset(val x: Reset) extends AnyVal {

    /** Poke a value onto the reset signal */
    def poke(value: Bool): Unit =
      toTestableReset(x).poke(value.asInstanceOf[Reset])
  }
}
