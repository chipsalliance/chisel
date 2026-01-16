// SPDX-License-Identifier: Apache-2.0

package chisel3

/** IO that reports the status of the test implemented by a testharness. */
private[chisel3] class SimulationTestHarnessIO extends Bundle with SimulationTestHarnessInterface {
  val clock = Input(Clock())
  val init = Input(Bool())
  val done = Output(Bool())
  val success = Output(Bool())
}

/** A fixed-IO module with the clock/init/done/success test interface that
 *  downstream test drivers such as ChiselSim and circt-test expect. Sets the
 *  implicit clock and reset. */
abstract class SimulationTestHarness
    extends FixedIORawModule(new SimulationTestHarnessIO)
    with SimulationTestHarnessInterface
    with ImplicitClock
    with ImplicitReset
    with Public {
  override final def clock = io.clock
  override final def init = io.init
  override final def done = io.done
  override final def success = io.success

  override def implicitClock: Clock = io.clock
  override def implicitReset: Reset = io.init
}
