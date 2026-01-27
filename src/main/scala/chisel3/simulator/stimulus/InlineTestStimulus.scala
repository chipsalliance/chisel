// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import scala.util.control.NoStackTrace

import chisel3.{Clock, Module, RawModule, Reset, SimulationTestHarnessInterface}
import chisel3.simulator.{AnySimulatedModule, Exceptions}
import chisel3.simulator.stimulus.{ResetProcedure, Stimulus}
import chisel3.experimental.inlinetest.{TestHarness => InlineTestHarness}

import firrtl.options.StageUtils.dramaticMessage

trait InlineTestStimulus extends Stimulus.Type[RawModule with SimulationTestHarnessInterface] {
  protected def _timeout: Int

  protected def _period: Int

  protected def _additionalResetCycles: Int

  private def applyImpl(dut: RawModule with SimulationTestHarnessInterface): Unit = {
    require(
      _period >= 2,
      s"specified period, '${_period}', must be 2 or greater because an integer half period must be non-zero"
    )

    val module = AnySimulatedModule.current
    val controller = module.controller

    val clock = module.port(dut.clock)
    val init = module.port(dut.init)
    val done = module.port(dut.done)
    val success = module.port(dut.success)

    ResetProcedure.testHarness(_additionalResetCycles, _period)(dut)

    clock.tick(
      timestepsPerPhase = _period / 2,
      maxCycles = _timeout,
      inPhaseValue = 1,
      outOfPhaseValue = 0,
      sentinel = Some(done, 1),
      checkElapsedCycleCount = { cycleCount =>
        if (cycleCount > _timeout) {
          throw new Exceptions.Timeout(_timeout, s"Test did not assert done before ${_timeout} cycles")
        }
      }
    )

    if (success.get().asBigInt == 0) {
      throw new Exceptions.TestFailed
    }
  }

  override final def apply(dut: RawModule with SimulationTestHarnessInterface): Unit =
    applyImpl(dut)

  final def apply(dut: InlineTestHarness[_]): Unit =
    applyImpl(dut)
}

object InlineTestStimulus {
  def apply(timeout: Int, additionalResetCycles: Int, period: Int) = new InlineTestStimulus {
    override val _timeout = timeout
    override val _period = period
    override val _additionalResetCycles = additionalResetCycles
  }
}
