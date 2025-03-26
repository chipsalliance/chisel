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

/** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
trait HasTests[M <: RawModule] { module: M =>

  /** Whether inline tests will be elaborated as a top-level definition to the circuit. */
  protected def elaborateTests: Boolean = true

  private val builderContext = internal.Builder.captureContext()

  private val thisModuleShouldElaborateTests =
    builderContext.includeInlineTestsForModule.exists { glob =>
      glob.replace("*", ".*").r.matches(desiredName)
    }

  private def shouldElaborateTest(testName: String) =
    builderContext.includeInlineTestsWithName.exists { glob =>
      glob.replace("*", ".*").r.matches(testName)
    }

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
    if (thisModuleShouldElaborateTests && elaborateTests && shouldElaborateTest(testName)) {
      elaborateParentModule { moduleDefinition =>
        val resetType = module match {
          case module: Module => Some(module.resetType)
          case _ => None
        }
        val test = new TestParameters[M, R](desiredName, testName, moduleDefinition, testBody, resetType)
        th.generate(test)
      }
    }
}
