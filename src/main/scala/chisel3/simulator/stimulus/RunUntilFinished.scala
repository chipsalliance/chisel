// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import chisel3.{Clock, Module, RawModule, SimulationTestHarnessInterface}
import chisel3.simulator.AnySimulatedModule
import chisel3.simulator.Exceptions

/** Stimulus that will run a simulation, expecting a [[chisel3.stop]] (a Verilog
  * `$finish`) to occur before a maximum number of cycles has elapsed.
  *
  * @see [[RunUntilSuccess]]
  */
trait RunUntilFinished[A] extends Stimulus.Type[A] {

  /** The maximum number of cycles. */
  protected def _maxCycles: Int

  /** A function that returns the clock to tick. */
  protected def _getClock: (A) => Clock

  /** The clock period in time precision units. */
  protected def _period: Int

  /** Apply stimulus to the unit
    *
    * @param the unit to apply stimulus to
    */
  override final def apply(dut: A): Unit = {
    require(
      _period >= 2,
      s"specified period, '${_period}', must be 2 or greater because an integer half period must be non-zero"
    )

    AnySimulatedModule.current
      .port(_getClock(dut))
      .tick(
        timestepsPerPhase = _period / 2,
        maxCycles = _maxCycles,
        inPhaseValue = 1,
        outOfPhaseValue = 0,
        sentinel = None,
        checkElapsedCycleCount = (count: BigInt) => {
          if (count == _maxCycles)
            throw new Exceptions.Timeout(_maxCycles, "Expected a $finish, but none received")
        }
      )
  }

}

/** Factory of [[RunUntilFinished]] stimulus for different kinds of modules. */
object RunUntilFinished {

  /** Return stimulus for a [[Module]].  This uses the default clock of the module to apply stimulus.
    *
    * @param maxCycles the maximum number of cycles to run the unit for before a timeout
    */
  def module(maxCycles: Int, period: Int = 10): RunUntilFinished[Module] = new RunUntilFinished[Module] {

    override protected final val _maxCycles = maxCycles

    override protected final val _getClock = _.clock

    override protected final val _period = period

  }

  /** Return stimulus for any type.  This requires the user to specify how to extract the clock from the type.
    * @param maxCycles the maximum number of cycles to run the unit for before a timeout
    * @param getClock a function to return a clock from the unit
    */
  def any[A](maxCycles: Int, getClock: A => Clock, period: Int = 10): RunUntilFinished[A] =
    new RunUntilFinished[A] {

      override protected final val _maxCycles = maxCycles

      override protected final val _getClock = getClock

      override protected final val _period = period

    }

  /** Return default stimulus.  This is the same as [[module]].
    *
    * @param maxCycles the maximum number of cycles to run the unit for before a timeout
    */
  def apply(maxCycles: Int, period: Int = 10): RunUntilFinished[Module] = {
    module(maxCycles, period)
  }

  /** Return stimulus for a [[SimulationTestHarnessInterface]].
    *
    * @param maxCycles the maximum number of cycles to run the unit for before a timeout
    */
  def testHarness[A <: RawModule with SimulationTestHarnessInterface](
    maxCycles: Int,
    period:    Int = 10
  ): RunUntilFinished[A] = any(
    maxCycles = maxCycles,
    getClock = _.clock,
    period = period
  )
}
