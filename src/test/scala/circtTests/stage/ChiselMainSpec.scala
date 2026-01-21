// SPDX-License-Identifier: Apache-2.0

package circtTests.stage

import chisel3._
import circt.stage.ChiselMain
import java.io.File
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.io.Source

object ChiselMainSpec {

  class ParameterlessModule extends RawModule {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))

    b :<= a
  }

  class BooleanModule(param: Boolean) extends RawModule {
    override final def desiredName = s"${this.getClass().getSimpleName()}_$param"
  }

  class IntegerModule(param: Int) extends RawModule {
    override final def desiredName = s"${this.getClass().getSimpleName()}_$param"
  }

  class DoubleModule(param: Double) extends RawModule {
    override final def desiredName = s"${this.getClass().getSimpleName()}_$param"
  }

  object Enum {
    sealed trait Type
    class A extends Type {
      override def toString = "A"
    }
    class B extends Type {
      override def toString = "B"
    }
  }
  class ObjectModule(param: Enum.Type) extends RawModule {
    override final def desiredName = s"${this.getClass().getSimpleName()}_$param"
  }

  class StringModule(param: String) extends RawModule {
    override final def desiredName = s"${this.getClass().getSimpleName()}_$param"
  }

  class MultipleParameters(bool: Boolean, int: Int) extends RawModule {
    override final def desiredName = s"${this.getClass().getSimpleName()}_${bool}_$int"
  }

}

class ChiselMainSpec extends AnyFunSpec with Matchers {
  import ChiselMainSpec._

  val testDir = new File("test_run_dir/ChiselMainSpec")
  case class Test(module: String, filename: String) {
    def test() = {
      val outFile = new File(testDir, filename)
      outFile.delete()
      outFile shouldNot exist

      info(module)
      ChiselMain.main(
        Array(
          "--module",
          module,
          "--target",
          "chirrtl",
          "--target-dir",
          testDir.toString
        )
      )

      outFile should exist
    }
  }

  describe("ChiselMain") {

    describe("support for modules without parameters") {

      it("should elaborate a parameterless module") {

        Test(
          "circtTests.stage.ChiselMainSpec$ParameterlessModule()",
          "ParameterlessModule.fir"
        ).test()

      }

    }

    describe("support for modules with parameters") {

      it("should elaborate a module with a Boolean parameter") {

        Test(
          "circtTests.stage.ChiselMainSpec$BooleanModule(true)",
          "BooleanModule_true.fir"
        ).test()

        Test(
          "circtTests.stage.ChiselMainSpec$BooleanModule(false)",
          "BooleanModule_false.fir"
        ).test()

      }

      it("should elaborate a module with an Integer parameter") {

        Test(
          "circtTests.stage.ChiselMainSpec$IntegerModule(42)",
          "IntegerModule_42.fir"
        ).test()

      }

      it("should elaborate a module with a Double parameter") {

        Test(
          "circtTests.stage.ChiselMainSpec$DoubleModule(3.141592654)",
          "DoubleModule_3141592654.fir"
        ).test()

      }

      it("should elaborate a module with an object parameter") {

        Test(
          "circtTests.stage.ChiselMainSpec$ObjectModule(circtTests.stage.ChiselMainSpec$Enum$A())",
          "ObjectModule_A.fir"
        ).test()

        Test(
          "circtTests.stage.ChiselMainSpec$ObjectModule(circtTests.stage.ChiselMainSpec$Enum$B())",
          "ObjectModule_B.fir"
        ).test()

      }

      it("should elaborate a module with a string parameter") {

        Test(
          """circtTests.stage.ChiselMainSpec$StringModule("hello")""",
          "StringModule_hello.fir"
        ).test()

      }

      it("should elaborate a module that takes multiple parameters") {

        Test(
          "circtTests.stage.ChiselMainSpec$MultipleParameters(true,42)",
          "MultipleParameters_true_42.fir"
        ).test()

      }

    }

  }

}
