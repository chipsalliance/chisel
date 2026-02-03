// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import chisel3.{RawModule, SimulationTestHarnessInterface}
import chisel3.experimental.inlinetest.{TestHarness => InlineTestHarness}

import firrtl.options.StageUtils.dramaticMessage

trait InlineTestStimulus extends Stimulus.Type[RawModule with SimulationTestHarnessInterface] {
  protected def _timeout: Int

  protected def _period: Int

  protected def _additionalResetCycles: Int

  override final def apply(dut: RawModule with SimulationTestHarnessInterface): Unit = {
    ResetProcedure.testHarness(_additionalResetCycles, _period)(dut)
    SimulationTestStimulus.testHarness(maxCycles = _timeout, period = _period)(dut)
  }

  final def apply(dut: InlineTestHarness[_]): Unit = {
    ResetProcedure.testHarness(_additionalResetCycles, _period)(dut)
    SimulationTestStimulus.testHarness(maxCycles = _timeout, period = _period)(dut)
  }
}

@deprecated("use SimulationTestStimulus instead", "Chisel 7.8.0")
object InlineTestStimulus {
  def apply(timeout: Int, additionalResetCycles: Int, period: Int) = new InlineTestStimulus {
    override val _timeout = timeout
    override val _period = period
    override val _additionalResetCycles = additionalResetCycles
  }
}
