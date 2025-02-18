// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import chisel3.{Clock, Module}
import chisel3.simulator.AnySimulatedModule
import chisel3.simulator.Exceptions

/** Stimulus that will run a simulation, expecting a [[chisel3.stop]] (a Verilog
  * `$finish`) to occur before a maximum number of cycles has elapsed.
  */
trait RunUntilFinished[A] extends Stimulus.Type[A] {

  /** The maximum number of cycles. */
  protected def _maxCycles: Int

  /** A function that returns the clock to tick. */
  protected def _getClock: (A) => Clock

  /** Apply stimulus to the unit
    *
    * @param the unit to apply stimulus to
    */
  override final def apply(dut: A): Unit = {
    AnySimulatedModule.current
      .port(_getClock(dut))
      .tick(
        timestepsPerPhase = 1,
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
  def module(maxCycles: Int): RunUntilFinished[Module] = new RunUntilFinished[Module] {

    override protected final val _maxCycles = maxCycles

    override protected final val _getClock = _.clock

  }

  /** Return stimulus for any type.  This requires the user to specify how to extract the clock from the type.
    * @param maxCycles the maximum number of cycles to run the unit for before a timeout
    * @param getClock a function to return a clock from the unit
    */
  def any[A](maxCycles: Int, getClock: A => Clock): RunUntilFinished[A] = new RunUntilFinished[A] {

    override protected final val _maxCycles = maxCycles

    override protected final val _getClock = getClock

  }

  /** Return default stimulus.  This is the same as [[module]].
    *
    * @param maxCycles the maximum number of cycles to run the unit for before a timeout
    */
  def apply(maxCycles: Int): RunUntilFinished[Module] = module(maxCycles)

}
