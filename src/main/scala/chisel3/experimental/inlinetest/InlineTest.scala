// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.inlinetest

import scala.collection.mutable

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.simulator.SimulationOutcome
import chisel3.util.Counter

import firrtl.options.StageUtils.dramaticMessage

object TestResult {

  /** The result of a test; i.e. simulation outcome vs. expectation. */
  sealed trait Type

  /** Outcome matched expectation. */
  case object Success extends Type

  /** Outcome did not match expectation. */
  case class Failure(message: String) extends Type
}

object TestBehavior {

  /** How a test should complete. */
  sealed trait Type {
    private[inlinetest] def getResult(actual:      SimulationOutcome.Type): TestResult.Type
    private[inlinetest] def driveInterface(result: TestHarnessInterface):   Unit
  }

  /** Test is expected to pass. */
  sealed trait ExpectsSuccess extends Type {
    def getResult(actual: SimulationOutcome.Type) =
      actual match {
        case SimulationOutcome.Success =>
          TestResult.Success
        case SimulationOutcome.Assertion(output) =>
          TestResult.Failure(output)
        case SimulationOutcome.Timeout(n) =>
          TestResult.Failure(s"timed out after ${n} cycles")
        case SimulationOutcome.SignaledFailure =>
          TestResult.Failure(s"test signaled failure")
      }
  }

  /** Test will run until someone calls `stop` or `finish`. */
  case object AwaitFinish extends ExpectsSuccess {
    def driveInterface(intf: TestHarnessInterface) = {
      intf.finish := 0.U
      intf.success := 1.U
    }
  }

  /** Test will run for `nCycles` cycles then signal success. */
  case class RunForCycles(nCycles: Int) extends ExpectsSuccess {
    def driveInterface(intf: TestHarnessInterface) = {
      val counter = Counter(nCycles)
      counter.inc()
      intf.finish := counter.value === (nCycles - 1).U
      intf.success := 1.U
    }
  }

  /** Test will finish when some condition is met. Test will succeed if some
   *  condition is met.
   */
  case class FinishWhen(finish: Bool, success: Bool) extends ExpectsSuccess {
    def driveInterface(intf: TestHarnessInterface) = {
      intf.finish := finish
      intf.success := success
    }
  }

  /** Test will finish when an assertion is raised and will check that the actual
   *  assertion matched what was expected.
   */
  case class ExpectAssertion private (isExpected: SimulationOutcome.Assertion => Boolean) extends Type {
    def getResult(actual: SimulationOutcome.Type) =
      actual match {
        case assertion: SimulationOutcome.Assertion if isExpected(assertion) =>
          TestResult.Success
        case assertion: SimulationOutcome.Assertion =>
          TestResult.Failure(s"unexpected assertion: ${assertion.simulatorOutput}")
        case SimulationOutcome.Timeout(n) =>
          TestResult.Failure(s"timed out after ${n} cycles")
        case SimulationOutcome.SignaledFailure =>
          TestResult.Failure(s"test signaled failure")
        case SimulationOutcome.Success =>
          TestResult.Failure(s"expected assertion, but simulation finished successfully")
      }
    def driveInterface(intf: TestHarnessInterface) = {
      intf.finish := 0.U
      intf.success := 1.U
    }
  }

  object ExpectAssertion {

    /** Test will finish when an assertion is raised and will check that the actual
     *  assertion contains the provided substring.
     */
    def contains(message: String) =
      ExpectAssertion { actual =>
        actual.simulatorOutput.contains(message)
      }
  }
}

/** A test, its expected behavior, and actual outcome. */
private[chisel3] class SimulatedTest private (
  val dutName:  String,
  val testName: String,
  testBehavior: TestBehavior.Type,
  outcome:      SimulationOutcome.Type
) {
  val result = testBehavior.getResult(outcome)
  val success = result match {
    case TestResult.Success => true
    case _                  => false
  }
}

object SimulatedTest {

  /** Construct an SimulatedTest from an ElaboratedTest by provided the outcome of the simulation. */
  private[chisel3] def apply(elaboratedTest: ElaboratedTest[_], outcome: SimulationOutcome.Type) =
    new SimulatedTest(
      elaboratedTest.dutName,
      elaboratedTest.testName,
      elaboratedTest.testBehavior,
      outcome
    )
}

object TestChoice {

  /** A choice of what test(s) to run. */
  sealed abstract class Type {
    private[chisel3] def globs: Seq[String]
    require(globs.nonEmpty, "Must provide at least one test to run")
  }

  /** Run tests matching any of these globs. */
  case class Globs(globs: Seq[String]) extends Type

  object Glob {

    /** Run tests matching a glob. */
    def apply(glob: String) = Globs(Seq(glob))
  }

  /** Run tests matching any of these names. */
  case class Names(names: Seq[String]) extends Type {
    override def globs = names
  }

  object Name {

    /** Run tests matching this name. */
    def apply(name: String) = Names(Seq(name))
  }

  /** Run all tests. */
  case object All extends Type {
    override def globs = Seq("*")
  }
}

/** Per-test parametrization needed to build a testharness that instantiates
  * the DUT and elaborates a test body.
  *
  *  @tparam M the type of the DUT module
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
private[chisel3] class TestHarnessInterface extends Bundle {

  /** The test shall be considered complete on the first positive edge of
   *  [[finish]] by the simulation. The [[TestHarness]] must drive this.
   */
  val finish = Bool()

  /** The test shall pass if this is asserted when the test is complete.
   *  The [[TestHarness]] must drive this.
   */
  val success = Bool()
}

/** TestHarnesses for inline tests should extend this. This abstract class sets the correct desiredName for
   *  the module, instantiates the DUT, and provides methods to generate the test. The [[resetType]] matches
   *  that of the DUT, or is [[Synchronous]] if it must be inferred.
   *
   *  @tparam M the type of the DUT module
   */
abstract class TestHarness[M <: RawModule](test: TestParameters[M])
    extends FixedIOModule(new TestHarnessInterface)
    with Public {
  override final def desiredName = test.testHarnessDesiredName
  override final def resetType = test.testHarnessResetType

  protected final val dut = Instance(test.dutDefinition())
  private[inlinetest] final val behavior = test.testBody(dut)

  behavior.driveInterface(io)
}

/** A test that has been elaborated to the circuit. */
private[chisel3] class ElaboratedTest[M <: RawModule] private (
  val dutName:     String,
  val testName:    String,
  val testHarness: TestHarness[M]
) {
  val testBehavior = testHarness.behavior
}

object ElaboratedTest {
  def apply[M <: RawModule](generator: TestHarnessGenerator[M], params: TestParameters[M]) =
    new ElaboratedTest(
      params.dutName(),
      params.testName,
      generator.generate(params)
    )
}

/** An implementation of a testharness generator. This is a type class that defines how to
 *  generate a testharness. It is passed to each invocation of [[HasTests.test]].
  *
  *  @tparam M the type of the DUT module
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
  val params: TestParameters[M] =
    new TestParameters(dutName, testName, dutDefinition, testBody, dutResetType)
  def elaborate() = ElaboratedTest(testHarnessGenerator, params)
}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
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
  private val elaboratedTests = new mutable.HashMap[String, ElaboratedTest[M]]

  /** Get the all tests elaborated to the ciruit. */
  private[chisel3] def getElaboratedTests: Seq[ElaboratedTest[M]] =
    elaboratedTests.values.toSeq

  /** Elaborate a test to the circuit as a public definition. */
  private def elaborateTestToCircuit[R](test: RegisteredTest[M]): Unit =
    Definition {
      val elaboratedTest = test.elaborate()
      elaboratedTests += test.params.testName -> elaboratedTest
      elaboratedTest.testHarness
    }

  /** Get all tests that were elaborated. */
  private[chisel3] def getElaboratedTestModules: Seq[ElaboratedTest[M]] =
    elaboratedTests.values.toSeq

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
