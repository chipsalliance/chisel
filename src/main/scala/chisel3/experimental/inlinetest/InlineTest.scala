// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.inlinetest

import scala.collection.mutable

import chisel3.{TestHarness => BaseTestHarness, TestHarnessInterface => BaseTestHarnessInterface, _}
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.simulator.SimulationOutcome
import chisel3.util.Counter

import firrtl.options.StageUtils.dramaticMessage
import chisel3.simulator.SimulationOutcome.Assertion

object TestResult {

  /** The result of a test; i.e. simulation outcome vs. expectation. */
  sealed trait Type

  /** Outcome matched expectation. */
  case object Success extends Type

  /** Outcome did not match expectation. */
  case class Failure(message: String) extends Type
}

final class TestConfiguration private (
  finishCondition:  Option[Bool],
  successCondition: Option[Bool],
  failureMessage:   Option[Printable]
) {
  private[inlinetest] def driveInterface(testName: String, intf: TestHarnessIO) = {
    intf.finish := finishCondition.getOrElse(false.B)
    intf.success := successCondition.getOrElse(true.B)
    failureMessage.foreach { failureMessage =>
      when(intf.finish && !intf.success) {
        printf(cf"${testName} failed: ${failureMessage}")
      }
    }
  }
}

object TestConfiguration {
  def default(): TestConfiguration =
    new TestConfiguration(finishCondition = None, successCondition = None, failureMessage = None)

  def runForCycles(nCycles: Int): TestConfiguration = {
    val (_, done) = Counter(true.B, nCycles)
    new TestConfiguration(
      finishCondition = Some(done),
      successCondition = None,
      failureMessage = None
    )
  }

  def apply(finish: Bool): TestConfiguration =
    new TestConfiguration(
      finishCondition = Some(finish),
      successCondition = None,
      failureMessage = None
    )

  def apply(finish: Bool, success: Bool, failureMessage: Printable): TestConfiguration =
    new TestConfiguration(
      finishCondition = Some(finish),
      successCondition = Some(success),
      failureMessage = Some(failureMessage)
    )
}

/** A test, its expected behavior, and actual outcome. */
private[chisel3] class SimulatedTest private (
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

  /** Construct an SimulatedTest from an ElaboratedTest by provided the outcome of the simulation. */
  private[chisel3] def apply(elaboratedTest: ElaboratedTest[_], outcome: SimulationOutcome.Type) =
    new SimulatedTest(
      elaboratedTest.dutName,
      elaboratedTest.testName,
      outcome
    )
}

object TestChoice {

  /** A choice of what test(s) to run. */
  sealed abstract class Type {
    private[chisel3] def globs: Array[String]
    require(globs.nonEmpty, "Must provide at least one test to run")
  }

  /** Run tests matching any of these globs. */
  case class Globs(private[chisel3] val globs: Array[String]) extends Type

  object Globs {
    def apply(globs: String*): Globs = new Globs(globs.toArray)
  }

  /** Run all tests. */
  case object All extends Type {
    override def globs = Array("*")
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
  private[inlinetest] val testBody: Instance[M] => TestConfiguration,
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

trait TestHarnessInterface extends BaseTestHarnessInterface

/** TestHarnesses for inline tests should extend this. This abstract class sets the correct desiredName for
   *  the module, instantiates the DUT, and provides methods to generate the test. The [[resetType]] matches
   *  that of the DUT, or is [[Synchronous]] if it must be inferred.
   *
   *  @tparam M the type of the DUT module
   */
abstract class TestHarness[M <: RawModule](test: TestParameters[M]) extends BaseTestHarness {
  override final def desiredName = test.testHarnessDesiredName

  override def implicitReset: Reset = test.testHarnessResetType match {
    case Module.ResetType.Asynchronous => io.init.asAsyncReset
    case _                             => io.init
  }

  final def resetType = test.testHarnessResetType
  final def reset = io.init

  protected final val dut = Instance(test.dutDefinition())
  private[inlinetest] final val testConfig = test.testBody(dut)

  testConfig.driveInterface(test.testName, io)
}

/** A test that has been elaborated to the circuit. */
private[chisel3] class ElaboratedTest[M <: RawModule] private (
  val dutName:     String,
  val testName:    String,
  val testHarness: TestHarness[M]
)

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
  testBody: Instance[M] => TestConfiguration,
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
  private val elaboratedTests = new mutable.LinkedHashMap[String, ElaboratedTest[M]]

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
  )(testBody: Instance[M] => TestConfiguration)(implicit testHarnessGenerator: TestHarnessGenerator[M]): Unit = {
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
