// SPDX-License-Identifier: Apache-2.0

package chisel3.autotest

import chisel3.experimental.inlinetest.ElaboratedTest
import chisel3.simulator.SimulationOutcome

/** An auto-test, its expected behavior, and actual outcome. */
private[chisel3] class SimulatedTest private[chisel3] (
  val dutName:  String,
  val testName: String,
  outcome:      SimulationOutcome.Type
) {
  val result = outcome match {
    case SimulationOutcome.Success                    => TestResult.Success
    case SimulationOutcome.Assertion(simulatorOutput) => TestResult.Failure(simulatorOutput)
    case SimulationOutcome.Timeout(n)                 => TestResult.Failure(s"timeout reached after ${n} timesteps")
    case SimulationOutcome.SignaledFailure            => TestResult.Failure(s"test signaled failure")
  }
  val success = result == TestResult.Success
}

object SimulatedTest {

  /** Construct a simulated auto-test from an elaborated test and its outcome. */
  private[chisel3] def apply(elaboratedTest: ElaboratedTest[_], outcome: SimulationOutcome.Type) =
    new SimulatedTest(
      elaboratedTest.dutName,
      elaboratedTest.testName,
      outcome
    )
}
