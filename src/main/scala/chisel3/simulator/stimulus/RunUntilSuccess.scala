// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import chisel3.{Bool, Clock, Module}
import chisel3.simulator.AnySimulatedModule
import chisel3.simulator.Exceptions

/** Stimulus that will run a simulation expecting a "success" port to assert.
  *
  * If the specified "success" port does not assert, then an
  * [[Exceptions.Timeout]] will be thrown.
  *
  * @see [[RunUntilFinished]]
  */
trait RunUntilSuccess[A] extends Stimulus.Type[A] {

  /** The maximum number of cycles. */
  protected def _maxCycles: Int

  /** A function that returns the clock to tick. */
  protected def _getClock: (A) => Clock

  /** A function that returns the success port. */
  protected def _getSuccess: (A) => Bool

  /** Apply stimulus to the unit
    *
    * @param the unit to apply stimulus to
    */
  override final def apply(dut: A): Unit = {
    val module = AnySimulatedModule.current
    val clock = module.port(_getClock(dut))
    val success = module.port(_getSuccess(dut))

    clock
      .tick(
        timestepsPerPhase = 1,
        maxCycles = _maxCycles,
        inPhaseValue = 1,
        outOfPhaseValue = 0,
        sentinel = Some((success, 1)),
        checkElapsedCycleCount = (count: BigInt) => {
          if (count == _maxCycles)
            throw new Exceptions.Timeout(_maxCycles, "Expected simulation to assert 'success' port")
        }
      )
  }

}

object RunUntilSuccess {

  /** Return stimulus for a [[Module]]
    *
    * @param maxCycles the maximum number of cycles to run the unit for before a
    * timeout
    * @param getSuccess a function to return a port which asserts when the
    * simulation has sucessfully finished
    */
  def module[A <: Module](maxCycles: Int, getSuccess: A => Bool): RunUntilSuccess[A] = new RunUntilSuccess[A] {

    override protected final val _maxCycles = maxCycles

    override protected final val _getClock = _.clock

    override protected final val _getSuccess = getSuccess

  }

  /** Return stimulus for any type.  This requires the user to specify how to
    * extract the clock and the success port from the type.
    *
    * @param maxCycles the maximum number of cycles to run the unit for before a
    * timeout
    * @param getClock a function to return a clock from the unit
    * @param getSuccess a function to return a port which asserts when the
    * simulation has sucessfully finished
    */
  def any[A](maxCycles: Int, getClock: A => Clock, getSuccess: A => Bool): RunUntilSuccess[A] = new RunUntilSuccess[A] {

    override protected final val _maxCycles = maxCycles

    override protected final val _getClock = getClock

    override protected final val _getSuccess = getSuccess

  }

  /** Return default stimulus.  This is the same as [[module]].
    *
    * @param maxCycles the maximum number of cycles to run the unit for before a
    * timeout
    * @param getSuccess a function to return a port which asserts when the
    * simulation has sucessfully finished
    */
  def apply[A <: Module](maxCycles: Int, getSuccess: A => Bool): RunUntilSuccess[A] = module(maxCycles, getSuccess)

}
