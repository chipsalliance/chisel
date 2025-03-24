// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import scala.collection.mutable

import chisel3._
import chisel3.experimental.hierarchy._

package object inlinetest {

  /** Per-test parametrization needed to build a testharness that instantiates
  * the DUT and elaborates a test body.
  *
  *  @tparam M the type of the DUT module
  *  @tparam R the type of the result returned by the test body
  */
  class TestHarnessParameters[M <: RawModule, R] private[inlinetest] (
    /** The [[desiredName]] of the DUT module. */
    val dutName: String,
    /** The user-provided name of the test. */
    val testName: String,
    /** Thunk for a Definition of the DUT module. */
    val dutDefinition: Definition[M],
    /** The body for this test, returns a result. */
    val testBody: Instance[M] => R,
    /** The reset type of the DUT module. */
    val resetType: Option[Module.ResetType.Type]
  ) {
    final def desiredTestModuleName = s"test_${dutName}_${testName}"
  }

  /** Everything we need to generate an inline test, except the DUT definition. */
  private[inlinetest] class TestGenerator[M <: RawModule, R](
    /** The [[desiredName]] of the DUT module. */
    dutName: String,
    /** The user-provided name of the test. */
    val testName: String,
    /** The body for this test, returns a result. */
    testBody: Instance[M] => R,
    /** The reset type of the DUT module. */
    resetType: Option[Module.ResetType.Type],
    /** Generator for the testharness module. */
    testHarnessGenerator: TestHarnessGenerator[M, R]
  ) {
    def generate(dutDefinition: Definition[M]): TestHarnessModule =
      testHarnessGenerator.generate(
        new TestHarnessParameters(dutName, testName, dutDefinition, testBody, resetType)
      )
  }

  private[inlinetest] object TestGenerator {
    def apply[M <: RawModule, R](
      dutName:              String,
      testName:             String,
      testBody:             Instance[M] => R,
      resetType:            Option[Module.ResetType.Type],
      testHarnessGenerator: TestHarnessGenerator[M, R]
    ): TestGenerator[M, R] = TestGenerator(
      dutName,
      testName,
      testBody,
      resetType,
      testHarnessGenerator
    )
  }

  trait TestHarnessInterface { this: Module with Public =>

    /** An output; shall be asserted when the test is finished. */
    def finish: Bool // output
    /** An output; shall be asserted when the test passes; sampled on the rising edge of [[finish]]. */
    def success: Bool // output
  }

  type TestHarnessModule = Module with TestHarnessInterface

  /** TestHarnesses for inline tests should extend this. This trait sets the correct desiredName for
 *  the module, instantiates the DUT, and provides methods to generate the test. By default, the
 *  reset is synchronous, but this can be changed by overriding [[resetType]].
 *
 *  @tparam M the type of the DUT module
 *  @tparam R the type of the result returned by the test body
 */
  class BaseTestHarness[M <: RawModule, R](test: TestHarnessParameters[M, R])
      extends Module
      with Public
      with TestHarnessInterface {
    this: RawModule =>
    override def desiredName = test.desiredTestModuleName

    override def resetType = test.resetType match {
      case Some(rt @ Module.ResetType.Synchronous)  => rt
      case Some(rt @ Module.ResetType.Asynchronous) => rt
      case _                                        => Module.ResetType.Synchronous
    }

    val finish = IO(Output(Bool()))
    val success = IO(Output(Bool()))

    // Handle the base case where a test has no result. In this case, we expect
    // the test to end the simulation and signal pass/fail.
    finish := false.B
    success := true.B

    protected val dut = Instance(test.dutDefinition)
    protected val testResult = test.testBody(dut)
  }

  /** An implementation of a testharness generator. This is a type class that defines how to
 *  generate a testharness. It is passed to each invocation of [[HasTests.test]].
  *
  *  @tparam M the type of the DUT module
  *  @tparam R the type of the result returned by the test body
  */
  trait TestHarnessGenerator[M <: RawModule, R] {

    /** Generate a testharness module given the test parameters. */
    def generate(test: TestHarnessParameters[M, R]): BaseTestHarness[M, R]
  }

  object TestHarnessGenerator {
    implicit def unitTestHarness[M <: RawModule]: TestHarnessGenerator[M, Unit] =
      new TestHarnessGenerator[M, Unit] {
        override def generate(test: TestHarnessParameters[M, Unit]) = new BaseTestHarness[M, Unit](test)
      }
  }

  /** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
  trait HasTests[M <: RawModule] { module: M =>

    private val globalElaborateTests =
      internal.Builder.captureContext().elaborateInlineTests

    /** Whether inline tests will be elaborated as a top-level definition to the circuit. */
    protected def elaborateTests: Boolean = true

    private val testGenerators = new mutable.HashMap[String, (Int, TestGenerator[M, _])]

    def elaborateTest(testName: String): TestHarnessModule =
      testGenerators
        .get(testName)
        .map(_._2)
        .getOrElse { throw new Exception(s"${this} has no test '${testName}'") }
        .generate(module.toDefinition)

    /** Generate a public module that instantiates this module. The default
    *  testharness has clock and synchronous reset IOs and contains the test
    *  body.
    *
    *  @param testBody the circuit to generate inside the testharness
    */
    protected final def test[R](
      testName: String
    )(testBody: Instance[M] => R)(implicit testHarnessGenerator: TestHarnessGenerator[M, R]): Unit = {
      val resetType = module match {
        case module: Module => Some(module.resetType)
        case _ => None
      }
      require(!testGenerators.contains(testName), s"test '${testName}' already declared")
      val testGenerator = new TestGenerator(module.desiredName, testName, testBody, resetType, testHarnessGenerator)
      testGenerators.addOne(testName -> (testGenerators.size, testGenerator))
    }

    afterModuleBuilt {
      if (elaborateTests && globalElaborateTests) {
        val moduleDefinition = module.toDefinition
        testGenerators.values.toSeq.sortBy(_._1).foreach { case (_, t) =>
          Definition(t.generate(moduleDefinition))
        }
      }
    }
  }
}
