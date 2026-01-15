// SPDX-License-Identifier: Apache-2.0

package chisel3

/** IO that reports the status of the test implemented by a testharness. */
class TestHarnessIO extends Bundle with TestHarnessInterface {

  /** The test driver shall drive a constant clock.
   */
  val clock = Input(Clock())

  /** The test driver shall assert init for one cycle to begin the test.
   */
  val init = Input(Bool())

  /** The test shall be considered complete on the first positive edge of
   *  [[finish]] by the simulation.
   */
  val finish = Output(Bool())

  /** The test shall pass if this is asserted when the test is complete.
   *  The [[TestHarness]] must drive this.
   */
  val success = Output(Bool())
}

/** A fixed-IO module with the clock/init/finish/success test interface that
 *  downstream test drivers such as ChiselSim and circt-test expect. Sets the
 *  implicit clock and reset. */
abstract class TestHarness
    extends FixedIORawModule(new TestHarnessIO)
    with TestHarnessInterface
    with ImplicitClock
    with ImplicitReset
    with Public {
  override final val clock = io.clock
  override final val init = io.init
  override final val finish = io.finish
  override final val success = io.success

  override def implicitClock: Clock = io.clock
  override def implicitReset: Reset = io.init
}
