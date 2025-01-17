package chisel3.experimental

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}

/** Provides methods to elaborate additional parents to the circuit. */
trait ElaboratesParents { module: RawModule =>
  private lazy val moduleDefinition =
    module.toDefinition.asInstanceOf[Definition[module.type]]

  /** Generate an additional parent around this module.
    *
    *  @param parent generator function, should instantiate the [[Definition]]
    */
  def elaborateParentModule(parent: Definition[module.type] => RawModule with Public): Unit =
    afterModuleBuilt { Definition(parent(moduleDefinition)) }
}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
  *
  *  @tparam TestResult the type returned from each test body generator, typically
  *  hardware indicating completion and/or exit code to the testharness.
  */
trait HasTestsWithResult[TestResult] extends ElaboratesParents { module: RawModule =>

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
    body:       Instance[module.type] => TestResult
  ): RawModule with Public = new Module with Public {
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
    body(dut.asInstanceOf[Instance[module.type]])
  }

  /** Generate a public module that instantiates this module. The default
    *  testharness has clock and synchronous reset IOs and contains the test
    *  body.
    *
    *  @param body the circuit to elaborate inside the testharness
    */
  final def test(name: String)(body: Instance[module.type] => TestResult): Unit = {
    elaborateParentModule {
      generateTestHarness(name, _, body)
    }
  }
}

/** Provides methods to build unit testharnesses inline after this module is elaborated.
 *  The test bodies do not communicate with the testharness and are expected to end the
 *  simulation themselves.
 */
trait HasTests extends HasTestsWithResult[Unit] { this: RawModule => }
