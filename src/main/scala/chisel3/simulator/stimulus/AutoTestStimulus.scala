// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import chisel3.{RawModule, SimulationTestHarnessInterface}
import chisel3.autotest.TestHarness

import scala.annotation.nowarn

/** Apply standard reset and timeout stimulus to an auto-test harness. */
trait AutoTestStimulus extends Stimulus.Type[RawModule with SimulationTestHarnessInterface] {
  protected def _timeout: Int

  protected def _period: Int

  protected def _additionalResetCycles: Int

  override final def apply(dut: RawModule with SimulationTestHarnessInterface): Unit =
    AutoTestStimulus.applyStimulus(dut, _timeout, _period, _additionalResetCycles)

  final def apply(dut: TestHarness[_]): Unit =
    AutoTestStimulus.applyStimulus(dut, _timeout, _period, _additionalResetCycles)
}

object AutoTestStimulus {

  /** Create auto-test stimulus with the given timeout and reset settings. */
  def apply(timeout: Int, additionalResetCycles: Int, period: Int) = new AutoTestStimulus {
    override val _timeout = timeout
    override val _period = period
    override val _additionalResetCycles = additionalResetCycles
  }

  private[chisel3] def applyStimulus(
    dut:                   RawModule with SimulationTestHarnessInterface,
    timeout:               Int,
    period:                Int,
    additionalResetCycles: Int
  ): Unit = {
    ResetProcedure.testHarness(additionalResetCycles, period)(dut)
    SimulationTestStimulus.testHarness(maxCycles = timeout, period = period)(dut)
  }
}

/** The old inline-test stimulus type. */
@deprecated("Use chisel3.simulator.stimulus.AutoTestStimulus", "Chisel 7.16.0")
trait InlineTestStimulus extends Stimulus.Type[RawModule with SimulationTestHarnessInterface] {
  protected def _timeout: Int

  protected def _period: Int

  protected def _additionalResetCycles: Int

  override final def apply(dut: RawModule with SimulationTestHarnessInterface): Unit =
    AutoTestStimulus.applyStimulus(dut, _timeout, _period, _additionalResetCycles)

  @nowarn("cat=deprecation") // accepts deprecated chisel3.experimental.inlinetest.TestHarness
  final def apply(dut: chisel3.experimental.inlinetest.TestHarness[_]): Unit =
    AutoTestStimulus.applyStimulus(dut, _timeout, _period, _additionalResetCycles)
}

/** The old inline-test stimulus factory. */
@deprecated("Use SimulationTestStimulus instead", "Chisel 7.8.0")
@nowarn("cat=deprecation") // returns deprecated InlineTestStimulus
object InlineTestStimulus {

  /** Create old inline-test stimulus. */
  def apply(timeout: Int, additionalResetCycles: Int, period: Int): InlineTestStimulus = new InlineTestStimulus {
    override val _timeout = timeout
    override val _period = period
    override val _additionalResetCycles = additionalResetCycles
  }
}
