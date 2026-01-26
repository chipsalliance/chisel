// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import chisel3.{Bool, Clock, Module, RawModule, SimulationTestHarnessInterface}
import chisel3.simulator.{AnySimulatedModule, Exceptions}

/** Stimulus that runs a simulation until a "done" signal asserts, then checks
  * a "success" signal to determine pass/fail.
  *
  * If the "done" signal does not assert before the maximum number of cycles,
  * an [[Exceptions.Timeout]] will be thrown. If "done" asserts but "success"
  * is not asserted, an [[Exceptions.TestFailed]] will be thrown.
  *
  * @see [[RunUntilSuccess]]
  * @see [[RunUntilFinished]]
  */
trait SimulationTestStimulus[A] extends Stimulus.Type[A] {

  /** The maximum number of cycles before timeout. */
  protected def _maxCycles: Int

  /** A function that returns the clock to tick. */
  protected def _getClock: A => Clock

  /** A function that returns the done signal. */
  protected def _getDone: A => Bool

  /** A function that returns the success signal. */
  protected def _getSuccess: A => Bool

  /** The clock period in time precision units. */
  protected def _period: Int

  /** Apply stimulus to the unit.
    *
    * @param dut the unit to apply stimulus to
    */
  override final def apply(dut: A): Unit = {
    require(
      _period >= 2,
      s"specified period, '${_period}', must be 2 or greater because an integer half period must be non-zero"
    )

    val module = AnySimulatedModule.current

    val clock = module.port(_getClock(dut))
    val done = module.port(_getDone(dut))
    val success = module.port(_getSuccess(dut))

    clock.tick(
      timestepsPerPhase = _period / 2,
      maxCycles = _maxCycles,
      inPhaseValue = 1,
      outOfPhaseValue = 0,
      sentinel = Some((done, 1)),
      checkElapsedCycleCount = { cycleCount =>
        if (cycleCount == _maxCycles) {
          throw new Exceptions.Timeout(_maxCycles, s"Test did not assert done before ${_maxCycles} cycles")
        }
      }
    )

    if (success.get().asBigInt == 0) {
      throw new Exceptions.TestFailed
    }
  }
}

object SimulationTestStimulus {

  /** Return stimulus for any type. This requires the user to specify how to
    * extract the clock, done, and success signals from the type.
    *
    * @param maxCycles the maximum number of cycles to run before timeout
    * @param getClock a function to return the clock from the unit
    * @param getDone a function to return the done signal from the unit
    * @param getSuccess a function to return the success signal from the unit
    * @param period the clock period in time precision units
    */
  def any[A](
    maxCycles:  Int,
    getClock:   A => Clock,
    getDone:    A => Bool,
    getSuccess: A => Bool,
    period:     Int = 10
  ): SimulationTestStimulus[A] = new SimulationTestStimulus[A] {
    override protected final val _maxCycles = maxCycles
    override protected final val _getClock = getClock
    override protected final val _getDone = getDone
    override protected final val _getSuccess = getSuccess
    override protected final val _period = period
  }

  /** Return stimulus for a [[Module]].
    *
    * @param maxCycles the maximum number of cycles to run before timeout
    * @param getDone a function to return the done signal from the module
    * @param getSuccess a function to return the success signal from the module
    * @param period the clock period in time precision units
    */
  def module[A <: Module](
    maxCycles:  Int,
    getDone:    A => Bool,
    getSuccess: A => Bool,
    period:     Int = 10
  ): SimulationTestStimulus[A] = any(
    maxCycles = maxCycles,
    getClock = _.clock,
    getDone = getDone,
    getSuccess = getSuccess,
    period = period
  )

  /** Return stimulus for a [[SimulationTestHarnessInterface]].
    *
    * @param maxCycles the maximum number of cycles to run before timeout
    * @param period the clock period in time precision units
    */
  def testHarness[A <: RawModule with SimulationTestHarnessInterface](
    maxCycles: Int,
    period:    Int = 10
  ): SimulationTestStimulus[A] = any(
    maxCycles = maxCycles,
    getClock = _.clock,
    getDone = _.done,
    getSuccess = _.success,
    period = period
  )

  /** Return default stimulus. This is the same as [[module]].
    *
    * @param maxCycles the maximum number of cycles to run before timeout
    * @param getDone a function to return the done signal from the module
    * @param getSuccess a function to return the success signal from the module
    * @param period the clock period in time precision units
    */
  def apply[A <: RawModule with SimulationTestHarnessInterface](
    maxCycles: Int,
    period:    Int = 10
  ): SimulationTestStimulus[A] = testHarness(maxCycles, period)
}
