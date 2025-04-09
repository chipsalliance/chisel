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
final class TestParameters[M <: RawModule, R] private[inlinetest] (
  /** The [[desiredName]] of the DUT module. */
  val dutName: String,
  /** The user-provided name of the test. */
  val testName: String,
  /** A Definition of the DUT module. */
  val dutDefinition: Definition[M],
  /** The body for this test, returns a result. */
  private[inlinetest] val testBody: Instance[M] => R,
  /** The reset type of the DUT module. */
  private[inlinetest] val dutResetType: Option[Module.ResetType.Type]
) {
  /** The concrete Chisel type of the reset port of the testharness module. */
  private[inlinetest] final def testHarnessResetType = dutResetType match {
    case Some(rt @ Module.ResetType.Synchronous)  => Bool()
    case Some(rt @ Module.ResetType.Asynchronous) => AsyncReset()
    case _                                        => Bool()
  }
  /** The [[desiredName]] for the testharness module. */
  private[inlinetest] final def desiredTestModuleName = s"test_${dutName}_${testName}"
}

class TestResultBundle extends Bundle {
  /** The test shall be considered complete on the first positive edge of
   *  [[finish]] by the simulation. The [[TestHarness]] must drive this.
   */
  val finish = Bool()
  /** The test shall pass if this is asserted when the test is complete.
   *  The [[TestHarness]] must drive this.
   */
  val success = Bool()
}

/** The interface between a testharness and the simulation driver. */
class TestHarnessBundle(resetType: Reset) extends TestResultBundle {
  /** The clock port shall be driven at a constant frequency by the simulation. */
  val clock = Flipped(Clock())
  /** The reset port shall be asserted for one cycle from the first positive edge
   *  of [[clock]] by the simulation.
   *  */
  val reset = Flipped(resetType)
}

/** TestHarnesses for inline tests should extend this. This abstract class sets the correct desiredName for
   *  the module, instantiates the DUT, and provides methods to generate the test. The [[resetType]] matches
   *  that of the DUT, or is [[Synchronous]] if it must be inferred (this can be overriden).
   *
   *  @tparam M the type of the DUT module
   *  @tparam R the type of the result returned by the test body
   */
abstract class TestHarness[M <: RawModule, R](test: TestParameters[M, R]) extends FixedIORawModule(new TestHarnessBundle(test.testHarnessResetType)) with Public {
  override final def desiredName = test.desiredTestModuleName

  // Handle the base case where a test has no result. In this case, we expect
  // the test to end the simulation and signal pass/fail.
  io.finish := false.B
  io.success := true.B

  protected final val dut = Instance(test.dutDefinition)
  protected final val testResult = test.testBody(dut)
}

/** TestHarnesses for inline tests should extend this. This abstract class sets the correct desiredName for
   *  the module, instantiates the DUT, and provides methods to generate the test. The [[resetType]] matches
   *  that of the DUT, or is [[Synchronous]] if it must be inferred (this can be overriden).
   *
   *  @tparam M the type of the DUT module
   *  @tparam R the type of the result returned by the test body
   */
abstract class TestHarnessWithResult[M <: RawModule](test: TestParameters[M, TestResultBundle]) extends TestHarness[M, TestResultBundle](test) {
  io.finish := testResult.finish
  io.success := testResult.success
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

  /** Provides a default testharness for tests that return [[Unit]]. */
  implicit def baseTestHarnessGenerator[M <: RawModule]: TestHarnessGenerator[M, Unit] = {
    TestHarnessGenerator(new TestHarness[M, Unit](_) {})
  }

  /** Provides a default testharness for tests that return a [[TestResultBundle]] */
  implicit def resultTestHarnessGenerator[M <: RawModule]: TestHarnessGenerator[M, TestResultBundle] = {
    TestHarnessGenerator(new TestHarnessWithResult[M](_) {})
  }
}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
trait HasTests { module: RawModule =>
  type M = module.type

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
      val harness = th.generate(test)
      harness.asInstanceOf[RawModule with Public]
    }
}
