// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.inlinetest

import scala.collection.mutable

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

/** Contains traits that implement behavior common to generators for unit test testharness modules. */
object TestHarness {
  import chisel3.{Module => ChiselModule, RawModule => ChiselRawModule}

  /** TestHarnesses for inline tests without clock and reset IOs should extend this. This
    * trait sets the correct desiredName for the module, instantiates the DUT, and provides
    * methods to elaborate the test.
    *
    *  @tparam M the type of the DUT module
    *  @tparam R the type of the result returned by the test body
    */
  trait RawModule[M <: ChiselRawModule, R] extends Public { this: ChiselRawModule =>
    def test: TestParameters[M, R]
    override def desiredName = test.desiredTestModuleName
    val dut = Instance(test.dutDefinition)
    final def elaborateTest(): R = test.testBody(dut)
  }

  /** TestHarnesses for inline tests should extend this. This trait sets the correct desiredName for
   *  the module, instantiates the DUT, and provides methods to elaborate the test. By default, the
   *  reset is synchronous, but this can be changed by overriding [[resetType]].
   *
   *  @tparam M the type of the DUT module
   *  @tparam R the type of the result returned by the test body
   */
  trait Module[M <: ChiselRawModule, R] extends RawModule[M, R] { this: ChiselModule =>
    override def resetType = test.resetType match {
      case Some(rt @ Module.ResetType.Synchronous)  => rt
      case Some(rt @ Module.ResetType.Asynchronous) => rt
      case _                                        => Module.ResetType.Synchronous
    }
  }
}

/** An implementation of a testharness generator. This is a type class that defines how to
 *  generate a testharness. It is passed to each invocation of [[HasTests.test]].
  *
  *  @tparam M the type of the DUT module
  *  @tparam R the type of the result returned by the test body
  */
trait TestHarnessGenerator[M <: RawModule, R] {

  /** Generate a testharness module given the test parameters. */
  def generate(test: TestParameters[M, R]): RawModule with Public
}

object TestHarnessGenerator {

  /** The minimal implementation of a unit testharness. Has a clock input and a synchronous reset
    *  input. Connects these to the DUT and does nothing else.
    */
  class UnitTestHarness[M <: RawModule](val test: TestParameters[M, Unit])
      extends Module
      with TestHarness.Module[M, Unit] {
    elaborateTest()
  }

  implicit def unitTestHarness[M <: RawModule]: TestHarnessGenerator[M, Unit] = new TestHarnessGenerator[M, Unit] {
    override def generate(test: TestParameters[M, Unit]) = new UnitTestHarness(test)
  }
}

private class TestGenerator[M <: RawModule, R](
  /** The index of this test in the order they were declared. */
  val index: Int,
  /** The user-provided name of the test. */
  val testName: String,
  /** The (eventually) legalized name for the DUT module */
  dutName: () => String,
  /** The body for this test, returns a result. */
  testBody: Instance[M] => R,
  /** The reset type of the DUT module. */
  resetType: Option[Module.ResetType.Type],
  /** The testharness generator. */
  testHarnessGenerator: TestHarnessGenerator[M, R]
) {
  def params(dutDefinition: Definition[M]) = new TestParameters(dutName(), testName, dutDefinition, testBody, resetType)
  def generate(dutDefinition: Definition[M]) = testHarnessGenerator.generate(params(dutDefinition))
}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
trait HasTests[M <: RawModule] { module: M =>

  /** Generators for inline tests by name. */
  private val testGenerators = new mutable.HashMap[String, TestGenerator[M, _]]

  lazy val moduleDefinition = module.toDefinition.asInstanceOf[Definition[M]]

  /** Get the tests that will be elaborated if tests are enabled for this module. */
  private def getTests: Seq[TestParameters[M, _]] =
    testGenerators.values.toSeq.sortBy(_.index).map(_.params(moduleDefinition))

  /** Call a function for each test after module elaboration.
   *
   *  @param fn function that takes the test parameters
   */
  def foreachTest(fn: TestParameters[M, _] => Unit): Unit =
    atModuleBodyEnd { getTests.foreach(fn) }

  /** Generate a public module that instantiates this module. The default
    *  testharness has clock and synchronous reset IOs and contains the test
    *  body.
    *
    *  @param testBody the circuit to elaborate inside the testharness
    */
  protected final def test[R](
    testName: String
  )(testBody: Instance[M] => R)(implicit testHarnessGenerator: TestHarnessGenerator[M, R]): Unit = {
    val resetType = module match {
      case module: Module => Some(module.resetType)
      case _ => None
    }
    require(!testGenerators.contains(testName), s"test '${testName}' already declared")
    val testGenerator =
      new TestGenerator(
        index = testGenerators.size,
        testName,
        () => module.name,
        testBody,
        resetType,
        testHarnessGenerator
      )
    testGenerators.addOne(testName -> testGenerator)
  }

  afterModuleBuilt {
    lazy val moduleDefinition = module.toDefinition.asInstanceOf[Definition[M]]
    testGenerators.values.toSeq.sortBy(_.index).foreach { t =>
      Definition(t.generate(moduleDefinition))
    }
  }
}
