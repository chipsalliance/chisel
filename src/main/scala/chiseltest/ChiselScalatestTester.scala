// SPDX-License-Identifier: Apache-2.0

package chiseltest

import chisel3._
import chisel3.simulator.EphemeralSimulator._
import chisel3.simulator.ChiselSim
import svsim._
import svsim.verilator.Backend.CompilationSettings.{TraceStyle, TraceKind}
import scala.language.implicitConversions

/**
 * ChiselTest-compatible API that delegates to ChiselSim (Chisel 7)
 *
 * This trait provides the ChiselTest API that users are familiar with from Chisel 6,
 * but internally uses ChiselSim from Chisel 7 to perform the actual testing.
 *
 * VCD GENERATION SUPPORT:
 * This compatibility layer supports VCD generation when WriteVcdAnnotation is used.
 * VCD files are generated in: build/chiselsim/<timestamp>/workdir-verilator/trace.vcd
 *
 * AUTOMATIC RESET FEATURE:
 * By default, this trait automatically resets the module before running tests,
 * mimicking ChiselTest's behavior from Chisel 6.
 *
 * To disable auto-reset or customize the reset duration:
 * {{{
 * class MyTest extends AnyFlatSpec with ChiselScalatestTester {
 *   override def autoResetEnabled: Boolean = false  // Disable auto-reset
 *   override def resetCycles: Int = 5               // Or change duration
 * }
 * }}}
 *
 * Example usage with VCD:
 * {{{
 * import chiseltest._
 * import org.scalatest.flatspec.AnyFlatSpec
 *
 * class MyModuleSpec extends AnyFlatSpec with ChiselScalatestTester {
 *   behavior of "MyModule"
 *
 *   it should "generate waveforms" in {
 *     test(new MyModule).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
 *       dut.io.in.poke(42.U)
 *       dut.clock.step()
 *       dut.io.out.expect(42.U)
 *     }
 *     // VCD file: build/chiselsim/<timestamp>/workdir-verilator/trace.vcd
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
   *
   * @param dutGen Generator for the device under test
   * @param autoReset Whether automatic reset is enabled
   * @param resetCyc Number of reset cycles to apply
   */
  class TestBuilder[T <: Module](dutGen: => T, autoReset: Boolean, resetCyc: Int) {
    /**
     * Attach ChiselTest-style annotations before running the test.
     *
     * @param annotations Annotation list (for example WriteVcdAnnotation)
     * @return A runner that executes the test with the provided annotations
     */
    def withAnnotations(annotations: Seq[Any]): TestRunner[T] = {
      new TestRunner(dutGen, autoReset, resetCyc, annotations)
    }

    /**
     * Run the test body without explicit annotations.
     *
     * @param body User-defined test body operating on the DUT instance
     */
    // Allow direct execution without annotations
    def apply(body: T => Unit): Unit = {
      val chiselSim = new ChiselSim {}
      chiselSim.simulate(dutGen) { dut =>
        if (autoReset) {
          applyReset(dut, resetCyc)
        }
        body(dut)
      }
    }
  }

  /**
   * Runner class that executes the test with VCD support
   *
   * @param dutGen Generator for the device under test
   * @param autoReset Whether automatic reset is enabled
   * @param resetCyc Number of reset cycles to apply
   * @param annotations Annotation list used to configure execution behavior
   */
  class TestRunner[T <: Module](dutGen: => T, autoReset: Boolean, resetCyc: Int, annotations: Seq[Any] = Seq()) {
    /**
     * Execute the configured test run.
     *
     * @param body User-defined test body operating on the DUT instance
     */
    def apply(body: T => Unit): Unit = {
      // Check if WriteVcdAnnotation is present
      val hasVcd = annotations.exists {
        case _: chiseltest.WriteVcdAnnotation.type => true
        case _ => false
      }

      val chiselSim = new ChiselSim {}

      if (hasVcd) {
        println("[ChiselTest Compat] WriteVcdAnnotation detected, enabling VCD trace generation...")
        
        // Create backend modification to enable VCD tracing
        implicit val backendMod: BackendSettingsModifications = 
          (settings: Backend.Settings) => settings match {
            case vs: verilator.Backend.CompilationSettings =>
              val vcdStyle = TraceStyle(
                kind = TraceKind.Vcd,
                traceUnderscore = false,
                traceStructs = true,
                traceParams = true,
                maxWidth = None,
                maxArraySize = None,
                traceDepth = None
              )
              vs.withTraceStyle(Some(vcdStyle))
            case other => other
          }
        
        // Simulate with VCD enabled
        chiselSim.simulate(dutGen) { dut =>
          chiselSim.enableWaves()
          
          if (autoReset) {
            applyReset(dut, resetCyc)
          }
          body(dut)
        }
      } else {
        // Simulate without VCD
        chiselSim.simulate(dutGen) { dut =>
          if (autoReset) {
            applyReset(dut, resetCyc)
          }
          body(dut)
        }
      }
    }
  }

  /**
   * Apply reset sequence to the DUT
   *
   * @param dut Device under test
   * @param cycles Number of cycles reset stays asserted
   */
  private def applyReset[T <: Module](dut: T, cycles: Int): Unit = {
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
   */
  implicit class testableData[T <: Data](val x: T) extends AnyVal {

    /**
     * Poke a value onto a port
     *
     * @param value The value to drive on the target signal
     */
    def poke(value: T): Unit = {
      toTestableData(x).poke(value)
    }

    /**
     * Peek the current value of a port
     *
     * @return The current sampled value of the target signal
     */
    def peek(): T = {
      toTestableData(x).peek()
    }

    /**
     * Assert that a port has an expected value
     *
     * @param expected The value expected on the target signal
     */
    def expect(expected: T): Unit = {
      toTestableData(x).expect(expected)
    }
  }

  /**
   * Implicit class to add ChiselTest-style operations to Clock types
   */
  implicit class testableClock(val clock: Clock) extends AnyVal {

    /**
     * Step the clock forward by a number of cycles
     *
     * @param cycles Number of clock cycles to advance (default: 1)
     */
    def step(cycles: Int = 1): Unit = {
      toTestableClock(clock).step(cycles)
    }
  }
}
