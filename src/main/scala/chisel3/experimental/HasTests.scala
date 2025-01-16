package chisel3.experimental

import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance}

object HasTests {
  case class Test[M <: RawModule](
    name:       String,
    definition: Definition[M],
    body:       Instance[M] => Unit,
  ) {
    lazy val instance     = Instance(definition)
    final def elaborate() = body(instance)
  }
}

/** Provides methods to build unit testharnesses inline after this module is elaborated. */
trait HasTests { this: RawModule =>
  import HasTests._

  private val module = this

  private lazy val moduleDefinition =
    module.toDefinition.asInstanceOf[Definition[this.type]]

  /** Given the specification for a test, generate the testharness module. Default
   *  testharness has clock and synchronous reset IOs and contains the test body.
   *
   *  @param test specification for the test
   */
  protected def genTestHarness(test: Test[this.type]): RawModule with Public =
    new Module with Public {
      override def desiredName = s"${module.desiredName}_test_${test.name}"
      override def resetType   = Module.ResetType.Synchronous
      test.elaborate()
    }

  /** Generate a public module that instantiates this module.
   *
   *  @param body the circuit to elaborate inside the testharness
   */
  def test(name: String)(body: Instance[this.type] => Unit): Unit = {
    afterModuleBuilt { Definition(genTestHarness(Test(name, moduleDefinition, body))) }
  }
}
