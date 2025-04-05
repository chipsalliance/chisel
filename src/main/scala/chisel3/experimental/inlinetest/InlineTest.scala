// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.inlinetest

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}

/** Per-test parametrization needed to build a testharness that instantiates
  * the DUT and elaborates a test body.
  *
  *  @tparam M the type of the DUT module
  *  @tparam R the type of the result returned by the test body
  */
class TestParameters[M <: RawModule, R] private[inlinetest] (
  /** The [[desiredName]] of the DUT module. */
  val dutName: String,
  /** The user-provided name of the test. */
  val testName: String,
  /** A Definition of the DUT module. */
  val dutDefinition: Definition[M],
  /** The body for this test, returns a result. */
  val testBody: Instance[M] => R,
  /** The reset type of the DUT module. */
  val resetType: Option[Module.ResetType.Type]
) {
  final def desiredTestModuleName = s"test_${dutName}_${testName}"
}

/** TestHarnesses for inline tests should extend this. This trait sets the correct desiredName for
   *  the module, instantiates the DUT, and provides methods to generate the test. By default, the
   *  reset is synchronous, but this can be changed by overriding [[resetType]].
   *
   *  @tparam M the type of the DUT module
   *  @tparam R the type of the result returned by the test body
   */
abstract class TestHarness[M <: RawModule, R](test: TestParameters[M, R]) extends Module with Public {
  override def desiredName = test.desiredTestModuleName

  override def resetType = test.resetType match {
    case Some(rt @ Module.ResetType.Synchronous)  => rt
    case Some(rt @ Module.ResetType.Asynchronous) => rt
    case _                                        => Module.ResetType.Synchronous
  }

  final val finish = IO(Output(Bool()))
  final val success = IO(Output(Bool()))

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
  def generate(test: TestParameters[M, R]): TestHarness[M, R]
}

object TestHarnessGenerator {

  /** Factory for a TestHarnessGenerator typeclass. */
  def apply[M <: RawModule, R](gen: TestParameters[M, R] => TestHarness[M, R]) =
    new TestHarnessGenerator[M, R] {
      override def generate(test: TestParameters[M, R]) = gen(test)
    }

  /** Typeclass for base [[TestHarness]] generator. */
  implicit def baseTestHarnessGenerator[M <: RawModule]: TestHarnessGenerator[M, Unit] = {
    TestHarnessGenerator(new TestHarness[M, Unit](_) {})
  }
}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
trait HasTests[M <: RawModule] { module: M =>

  /** A Definition of the DUT to be used for each of the tests. */
  private lazy val moduleDefinition =
    module.toDefinition.asInstanceOf[Definition[module.type]]

  /** Generate an additional parent around this module.
    *
    *  @param parent generator function, should instantiate the [[Definition]]
    */
  protected final def elaborateParentModule(parent: Definition[module.type] => RawModule with Public): Unit =
    afterModuleBuilt { Definition(parent(moduleDefinition)) }

  /** Generate a public module that instantiates this module. The default
    *  testharness has clock and synchronous reset IOs and contains the test
    *  body.
    *
    *  @param testBody the circuit to elaborate inside the testharness
    */
  protected final def test[R](
    testName: String
  )(testBody: Instance[M] => R)(implicit th: TestHarnessGenerator[M, R]): Unit =
    elaborateParentModule { moduleDefinition =>
      val resetType = module match {
        case module: Module => Some(module.resetType)
        case _ => None
      }
      val test = new TestParameters[M, R](desiredName, testName, moduleDefinition, testBody, resetType)
      th.generate(test)
    }
}
