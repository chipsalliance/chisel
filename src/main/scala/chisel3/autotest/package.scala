// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.annotation.nowarn

/** Provides the auto-test API. */
package object autotest {

  /** The result of an auto-test. */
  @nowarn("cat=deprecation")
  type TestResult = experimental.inlinetest.TestResult.type
  @nowarn("cat=deprecation")
  val TestResult = experimental.inlinetest.TestResult

  /** A configuration for an auto-test. */
  @nowarn("cat=deprecation")
  type TestConfiguration = experimental.inlinetest.TestConfiguration
  @nowarn("cat=deprecation")
  val TestConfiguration = experimental.inlinetest.TestConfiguration

  /** A choice of which auto-tests to run. */
  @nowarn("cat=deprecation")
  type TestChoice = experimental.inlinetest.TestChoice.type
  @nowarn("cat=deprecation")
  val TestChoice = experimental.inlinetest.TestChoice

  /** Per-test parameters used to build an auto-test harness. */
  @nowarn("cat=deprecation")
  type TestParameters[M <: RawModule] = experimental.inlinetest.TestParameters[M]

  /** The interface for an auto-test harness. */
  @nowarn("cat=deprecation")
  type TestHarnessInterface = experimental.inlinetest.TestHarnessInterface

  /** A harness that instantiates a DUT and runs an auto-test. */
  @nowarn("cat=deprecation")
  type TestHarness[M <: RawModule] = experimental.inlinetest.TestHarness[M]

  /** A type class that generates auto-test harnesses. */
  @nowarn("cat=deprecation")
  type TestHarnessGenerator[M <: RawModule] = experimental.inlinetest.TestHarnessGenerator[M]

  /** Create auto-test harness generators. */
  object TestHarnessGenerator {
    @nowarn("cat=deprecation")
    def apply[M <: RawModule](gen: TestParameters[M] => TestHarness[M]): TestHarnessGenerator[M] =
      experimental.inlinetest.TestHarnessGenerator(gen)

    @nowarn("cat=deprecation")
    implicit def baseTestHarnessGenerator[M <: RawModule]: TestHarnessGenerator[M] =
      experimental.inlinetest.TestHarnessGenerator.baseTestHarnessGenerator
  }

  /** Provides methods to build auto-test harnesses during module elaboration. */
  @nowarn("cat=deprecation")
  type HasTests = experimental.inlinetest.HasTests
}
