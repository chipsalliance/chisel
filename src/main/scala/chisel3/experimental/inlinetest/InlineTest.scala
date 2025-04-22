// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.inlinetest

import scala.collection.mutable
import scala.util.matching.Regex

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.simulator.{stimulus, Simulator}

/** Per-test parametrization needed to build a testharness that instantiates
  * the DUT and elaborates a test body.
  *
  *  @tparam M the type of the DUT module
  *  @tparam R the type of the result returned by the test body
  */
final class TestParameters[M <: RawModule] private[inlinetest] (
  /** The [[name]] of the DUT module. */
  private[inlinetest] val dutName: () => String,
  /** The user-provided name of the test. */
  val testName: String,
  /** A Definition of the DUT module. */
  private[inlinetest] val dutDefinition: () => Definition[M],
  /** The body for this test, returns a result. */
  private[inlinetest] val testBody: Instance[M] => TestBehavior.Type,
  /** The reset type of the DUT module. */
  private[inlinetest] val dutResetType: Option[Module.ResetType.Type]
  /** The expected result when simulating this test. */
) {

  /** The concrete reset type of the testharness module. */
  private[inlinetest] final def testHarnessResetType = dutResetType match {
    case Some(rt @ Module.ResetType.Synchronous)  => rt
    case Some(rt @ Module.ResetType.Asynchronous) => rt
    case _                                        => Module.ResetType.Synchronous
  }

  /** The [[desiredName]] for the testharness module. */
  def testHarnessDesiredName = s"test_${dutName()}_${testName}"
}

/** IO that reports the status of the test implemented by a testharness. */
final class TestResultBundle extends Bundle {

  /** The test shall be considered complete on the first positive edge of
   *  [[finish]] by the simulation. The [[TestHarness]] must drive this.
   */
  val finish = Bool()

  /** The test shall pass if this is asserted when the test is complete.
   *  The [[TestHarness]] must drive this.
   */
  val success = Bool()
}

object TestBehavior {

  /** How a test should complete. */
  trait Type {
    /** Drive the testharness interface (finish, success) in a way that is appropriate for this behavior. */
    def driveInterface(result: TestResultBundle):   Unit
  }

  /** Test will run until the DUT or test body calls [[chisel3.stop]] or `$finish`. Note: whatever is simulating
   *  this test (e.g. ChiselSim) may still enforce a timeout; reaching that likely still constitutes test failure.
   */
  case object RunForever extends Type {
    def driveInterface(intf: TestResultBundle) = {
      intf.finish := 0.U
      intf.success := 1.U
    }
  }

  /** Test will run for `nCycles` cycles then signal success. The test will only fail if an assertion or `$fatal` is raised. */
  case class RunForCycles(nCycles: Int) extends Type {
    def driveInterface(intf: TestResultBundle) = {
      val counter = chisel3.util.Counter(nCycles)
      counter.inc()
      intf.finish := counter.value === (nCycles - 1).U
      intf.success := 1.U
    }
  }

  /** Test will finish when some condition is met. Test will succeed if some condition is met. */
  case class FinishWhen(finish: Bool, success: Bool) extends Type {
    def driveInterface(intf: TestResultBundle) = {
      intf.finish := finish
      intf.success := success
    }
  }
}

/** TestHarnesses for inline tests should extend this. This abstract class sets the correct desiredName for
   *  the module, instantiates the DUT, and provides methods to generate the test. The [[resetType]] matches
   *  that of the DUT, or is [[Synchronous]] if it must be inferred (this can be overriden).
   *
   *  A [[TestHarness]] has the following ports:
   *
   * - [[clock]]: shall be driven at a constant frequency by the simulation.
   * - [[reset]]: shall be asserted for one cycle from the first positive edge of [[clock]] by the simulation.
   * - [[reset]]: shall be asserted for one cycle from the first positive edge of [[clock]] by the simulation.
   * - [[finish]]: the test shall be considered complete on the first positive edge of [[finish]].
   *
   *  @tparam M the type of the DUT module
   */
abstract class TestHarness[M <: RawModule](test: TestParameters[M])
    extends FixedIOModule(new TestResultBundle)
    with Public {
  override final def desiredName = test.testHarnessDesiredName
  override final def resetType = test.testHarnessResetType

  protected final val dut = Instance(test.dutDefinition())
  protected final val testResult = test.testBody(dut)

  testResult.driveInterface(io)
}

/** A test that has been elaborated to the circuit. */
private[chisel3] case class ElaboratedTest[M <: RawModule](
  params:      TestParameters[M],
  testHarness: TestHarness[M]
)

/** An implementation of a testharness generator. This is a type class that defines how to
 *  generate a testharness. It is passed to each invocation of [[HasTests.test]].
  *
  *  @tparam M the type of the DUT module
  *  @tparam R the type of the result returned by the test body
  */
trait TestHarnessGenerator[M <: RawModule] {

  /** Generate a testharness module given the test parameters. */
  def generate(test: TestParameters[M]): TestHarness[M]
}

object TestHarnessGenerator {

  /** Factory for a TestHarnessGenerator typeclass. */
  def apply[M <: RawModule](gen: TestParameters[M] => TestHarness[M]) =
    new TestHarnessGenerator[M] {
      override def generate(test: TestParameters[M]) = gen(test)
    }

  /** Provides a default testharness for tests that return [[Unit]]. */
  implicit def baseTestHarnessGenerator[M <: RawModule]: TestHarnessGenerator[M] = {
    TestHarnessGenerator(new TestHarness[M](_) {})
  }
}

/** A test that was registered, but is not necessarily selected for elaboration. */
private final class RegisteredTest[M <: RawModule](
  /** The user-provided name of the test. */
  val testName: String,
  /** Whether or not this test should be elaborated. */
  val shouldElaborateToCircuit: Boolean,
  /** Thunk that returns a [[Definition]] of the DUT */
  dutDefinition: () => Definition[M],
  /** The (eventually) legalized name for the DUT module */
  dutName: () => String,
  /** The body for this test, returns a result. */
  testBody: Instance[M] => TestBehavior.Type,
  /** The reset type of the DUT module. */
  dutResetType: Option[Module.ResetType.Type],
  /** The testharness generator. */
  testHarnessGenerator: TestHarnessGenerator[M]
) {
  val params: TestParameters[M] = new TestParameters(dutName, testName, dutDefinition, testBody, dutResetType)
  def elaborate() = ElaboratedTest(
    params,
    testHarnessGenerator.generate(params)
  )
}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
trait HasTests { module: RawModule =>
  private type M = module.type

  /** Whether inline tests will be elaborated as a top-level definition to the circuit. */
  protected def elaborateTests: Boolean = true

  /** From options, what modules and tests should be included in this run. */
  private val inlineTestIncluder = internal.Builder.captureContext().inlineTestIncluder

  /** Whether a test is enabled. */
  private def shouldElaborateTest(testName: String): Boolean =
    elaborateTests && inlineTestIncluder.shouldElaborateTest(module.desiredName, testName)

  /** This module as a definition. Lazy in order to prevent evaluation unless used by a test. */
  private lazy val moduleDefinition = module.toDefinition.asInstanceOf[Definition[M]]

  /** Generators for inline tests by name. LinkedHashMap preserves test insertion order. */
  private val registeredTests = new mutable.LinkedHashMap[String, RegisteredTest[M]]

  /** Get the currently registered tests for this module and whether they are queued for elaboration. */
  private def getRegisteredTests: Seq[RegisteredTest[M]] =
    registeredTests.values.toSeq

  /** Get all enabled tests for this module. */
  def getTests: Seq[TestParameters[M]] =
    getRegisteredTests.filter(_.shouldElaborateToCircuit).map(_.params)

  /** Map from test name to elaborated test. */
  private val elaboratedTests = new mutable.LinkedHashMap[String, ElaboratedTest[M]]

  /** Get the all tests elaborated to the ciruit. */
  private[chisel3] def getElaboratedTests: Seq[ElaboratedTest[M]] =
    elaboratedTests.values.toSeq

  /** Elaborate a test to the circuit as a public definition. */
  private def elaborateTestToCircuit(test: RegisteredTest[M]): Unit =
    Definition {
      val elaboratedTest = test.elaborate()
      elaboratedTests += test.params.testName -> elaboratedTest
      elaboratedTest.testHarness
    }

  /** Generate a public module that instantiates this module. The default
    *  testharness has clock and synchronous reset IOs and contains the test
    *  body.
    *
    *  @param testBody the circuit to elaborate inside the testharness
    */
  protected final def test(
    testName: String
  )(testBody: Instance[M] => TestBehavior.Type)(implicit testHarnessGenerator: TestHarnessGenerator[M]): Unit = {
    require(!registeredTests.contains(testName), s"test '${testName}' already declared")
    val dutResetType = module match {
      case module: Module => Some(module.resetType)
      case _ => None
    }
    val test =
      new RegisteredTest(
        testName,
        module.shouldElaborateTest(testName),
        () => moduleDefinition,
        () => module.name,
        testBody,
        dutResetType,
        testHarnessGenerator
      )
    registeredTests += testName -> test
  }

  afterModuleBuilt {
    getRegisteredTests.foreach { test =>
      if (test.shouldElaborateToCircuit) {
        elaborateTestToCircuit(test)
      }
    }
  }
}
