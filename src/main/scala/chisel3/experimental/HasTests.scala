package chisel3.experimental

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
trait HasTests[TestResult] { module: RawModule =>

  /** Type for a testharness module. */
  protected final type TestHarness = RawModule with Public

  /** Type for a test circuit generator. */
  protected final type TestBody = Instance[module.type] => TestResult

  /** A Definition of the DUT to be used for each of the tests. */
  private lazy val moduleDefinition =
    module.toDefinition.asInstanceOf[Definition[module.type]]

  /** Generate an additional parent around this module.
    *
    *  @param parent generator function, should instantiate the [[Definition]]
    */
  protected final def elaborateParentModule(parent: Definition[module.type] => RawModule with Public): Unit =
    afterModuleBuilt { Definition(parent(moduleDefinition)) }

  /** Given a [[Definition]] of this module, and a test body that takes an [[Instance]],
    *  generate a testharness module that implements the test.
    *
    *  The default behavior is a module with the following features:
    *
    *  - A clock port
    *  - A reset port with this module's reset type, or synchronous if unspecified
    *  - The [[desiredName]] is "[[this.desiredName]] _ [[name]]".
    */
  protected def generateTestHarness(
    testName:   String,
    definition: Definition[module.type],
    body:       TestBody
  ): TestHarness =
    new Module with Public {
      override def resetType = module match {
        case module: Module =>
          module.resetType match {
            case t @ Module.ResetType.Asynchronous => t
            case t @ Module.ResetType.Synchronous  => t
            case _                                 => Module.ResetType.Synchronous
          }
        case _ => Module.ResetType.Synchronous
      }
      override val desiredName = s"${module.desiredName}_${testName}"
      val dut = Instance(definition)
      (body(dut.asInstanceOf[Instance[module.type]]), ())
    }

  /** Generate a public module that instantiates this module. The default
    *  testharness has clock and synchronous reset IOs and contains the test
    *  body.
    *
    *  @param body the circuit to elaborate inside the testharness
    */
  protected final def test(name: String)(body: TestBody): Unit = {
    elaborateParentModule {
      generateTestHarness(name, _, body)
    }
  }
}
