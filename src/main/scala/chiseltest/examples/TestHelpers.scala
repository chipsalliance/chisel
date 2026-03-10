// SPDX-License-Identifier: Apache-2.0

package chiseltest.examples

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/**
 * Utility trait for common test setup patterns
 */
trait ChiselTestHelper extends AnyFlatSpec with ChiselScalatestTester {

  /**
   * Test a module with a basic poke and expect pattern
   */
  protected def testPokeExpect[T <: Module](
    module:        => T,
    pokeData:      (T, UInt) => Unit,
    expectedData:  UInt
  ): Unit = {
    test(module) { dut =>
      implicit val clk = dut.clock
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      pokeData(dut, 42.U)
      dut.clock.step(1)
    }
  }

  /**
   * Test a sequential logic module with multiple cycles
   */
  protected def testSequential[T <: Module](
    module:      => T,
    stimuli:     Seq[(T) => Unit]
  ): Unit = {
    test(module) { dut =>
      implicit val clk = dut.clock
      dut.reset.poke(true.B)
      dut.clock.step(2)
      dut.reset.poke(false.B)
      dut.clock.step(1)

      stimuli.foreach { stimulus =>
        stimulus(dut)
        dut.clock.step(1)
      }
    }
  }
}
