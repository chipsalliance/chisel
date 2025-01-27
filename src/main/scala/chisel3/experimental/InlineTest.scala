package chisel3.experimental

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}

package object inlinetest {

  /** Per-test parametrization needed to build a testharness that instantiates
    *  the DUT and elaborates a test body.
    *
    *  @tparam M the type of the DUT module
    *  @tparam R the type of the result returned by the test body
    */
  case class TestParameters[M <: RawModule, R](
    /** The [[desiredName]] of the DUT module. */
    dutName: String,
    /** The user-provided name of the test. */
    testName: String,
    /** A Definition of the DUT module. */
    dutDefinition: Definition[M],
    /** The body for this test, returns a result. */
    body: Instance[M] => R)

  /** An implementation of a testharness generator.
    *
    *  @tparam M the type of the DUT module
    *  @tparam R the type of the result returned by the test body
    */
  trait TestHarness[M <: RawModule, R] {

    /** Generate a testharness module given the test parameters. */
    def generate(test: TestParameters[M, R]): RawModule with Public
  }

  object TestHarness {

    /** The minimal implementation of a unit testharness. Has a clock input and a synchronous reset
      *  input. Connects these to the DUT and does nothing else.
      */
    class UnitTestHarness[M <: RawModule](test: TestParameters[M, Unit]) extends Module with Public {
      override def resetType = Module.ResetType.Synchronous
      override val desiredName = s"test_${test.dutName}_${test.testName}"
      val dut = Instance(test.dutDefinition)
      test.body(dut)
    }

    implicit def unitTestHarness[M <: RawModule]: TestHarness[M, Unit] = new TestHarness[M, Unit] {
      def generate(test: TestParameters[M, Unit]): RawModule with Public = new UnitTestHarness(test)
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
      *  @param body the circuit to elaborate inside the testharness
      */
    protected final def test[R](testName: String)(body: Instance[M] => R)(implicit th: TestHarness[M, R]): Unit =
      elaborateParentModule { moduleDefinition =>
        val test = TestParameters[M, R](desiredName, testName, moduleDefinition, body)
        th.generate(test)
      }
  }
}
