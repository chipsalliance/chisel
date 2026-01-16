// SPDX-License-Identifier: Apache-2.0

package chisel3

/** IO that reports the status of the test implemented by a testharness. */
trait SimulationTestHarnessInterface {

  /** The test driver shall drive a fixed clock.
   */
  def clock: Clock

  /** The test driver shall assert init for one cycle to begin the test.
   */
  def init: Bool

  /** The test shall be considered complete on the first positive pulse of
   *  [[done]] by the simulation.
   */
  def done: Bool

  /** The test shall pass if this is asserted when the test is complete.
   */
  def success: Bool
}
