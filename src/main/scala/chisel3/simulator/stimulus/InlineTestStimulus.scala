// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import scala.util.control.NoStackTrace

import chisel3.{Clock, Module, RawModule, Reset}
import chisel3.simulator.{AnySimulatedModule, Exceptions}
import chisel3.simulator.stimulus.{ResetProcedure, Stimulus}
import chisel3.experimental.inlinetest.TestHarness

import firrtl.options.StageUtils.dramaticMessage

trait InlineTestStimulus extends Stimulus.Type[TestHarness[_]] {
  protected def _timeout: Int

  protected def _period: Int

  override final def apply(dut: TestHarness[_]): Unit = {
    val module = AnySimulatedModule.current
    val controller = module.controller

    val clock = module.port(dut.clock)
    val reset = module.port(dut.reset)
    val finish = module.port(dut.io.finish)
    val success = module.port(dut.io.success)

    ResetProcedure.module(_period)(dut)

    var cycleCount = 0

    while (finish.get().asBigInt == 0) {
      clock.tick(
        timestepsPerPhase = 1,
        maxCycles = 1,
        inPhaseValue = 1,
        outOfPhaseValue = 0,
        sentinel = Some(finish, 1),
        checkElapsedCycleCount = { cycleCount =>
          if (cycleCount > _timeout) {
            throw new Exceptions.Timeout(_timeout, s"Test did not assert finish before ${_timeout} timesteps")
          }
        }
      )
    }

    if (success.get().asBigInt == 0) {
      throw new Exceptions.TestFailed
    }
  }
}

object InlineTestStimulus {
  def apply(timeout: Int, period: Int) = new InlineTestStimulus {
    override val _timeout = timeout
    override val _period = period
  }
}
