// SPDX-License-Identifier: Apache-2.0

package chisel3

/** IO that reports the status of the test implemented by a testharness. */
private[chisel3] trait TestHarnessInterface {

  /** The test driver shall drive a constant clock.
   */
  def clock: Clock

  /** The test driver shall assert init for one cycle to begin the test.
   */
  def init: Bool

  /** The test shall be considered complete on the first positive edge of
   *  [[finish]] by the simulation.
   */
  def finish: Bool

  /** The test shall pass if this is asserted when the test is complete.
   *  The [[TestHarness]] must drive this.
   */
  def success: Bool
}
