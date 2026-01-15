// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import scala.util.control.NoStackTrace

import chisel3.{Clock, Module, RawModule, Reset, TestHarness}
import chisel3.simulator.{AnySimulatedModule, Exceptions}
import chisel3.simulator.stimulus.{ResetProcedure, Stimulus}
import chisel3.experimental.inlinetest.{TestHarness => InlineTestHarness}

import firrtl.options.StageUtils.dramaticMessage

trait InlineTestStimulus extends Stimulus.Type[TestHarness] {
  protected def _timeout: Int

  protected def _period: Int

  protected def _additionalResetCycles: Int

  private def applyImpl(dut: TestHarness): Unit = {
    val module = AnySimulatedModule.current
    val controller = module.controller

    val clock = module.port(dut.io.clock)
    val init = module.port(dut.io.init)
    val finish = module.port(dut.io.finish)
    val success = module.port(dut.io.success)

    ResetProcedure.testHarness(_additionalResetCycles, _period)(dut)

    clock.tick(
      timestepsPerPhase = 1,
      maxCycles = _timeout,
      inPhaseValue = 1,
      outOfPhaseValue = 0,
      sentinel = Some(finish, 1),
      checkElapsedCycleCount = { cycleCount =>
        if (cycleCount > _timeout) {
          throw new Exceptions.Timeout(_timeout, s"Test did not assert finish before ${_timeout} timesteps")
        }
      }
    )

    if (success.get().asBigInt == 0) {
      throw new Exceptions.TestFailed
    }
  }

  override final def apply(dut: TestHarness): Unit =
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
