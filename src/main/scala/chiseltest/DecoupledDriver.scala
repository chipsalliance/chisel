// SPDX-License-Identifier: Apache-2.0

package chiseltest

/**
 * DecoupledDriver provides utility functions for testing Decoupled interfaces
 *
 * This is compatible with ChiselTest's DecoupledDriver but implemented using ChiselSim.
 *
 * The actual implementation is in the package object as implicit conversions.
 * Import chiseltest._ or chiseltest.DecoupledDriver._ to use these methods.
 *
 * Example:
 * {{{
 * import chiseltest._
 *
 * test(new QueueModule) { dut =>
 *   dut.io.in.enqueueNow(42.U)
 *   dut.io.out.expectDequeueNow(42.U)
 * }
 * }}}
 */
object DecoupledDriver {
  // For compatibility with imports like: import chiseltest.DecoupledDriver._
  // The actual implicit classes are defined in the package object
}
