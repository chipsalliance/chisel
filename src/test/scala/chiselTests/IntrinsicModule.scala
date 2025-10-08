// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental._
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.annotation.nowarn

class IntModuleTest extends IntrinsicModule("TestIntrinsic") {
  val foo = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

class IntModuleParam(str: String, dbl: Double)
    extends IntrinsicModule("OtherIntrinsic", Map("STRING" -> str, "REAL" -> dbl)) {}

class IntModuleGenName(GenIntName: String) extends IntrinsicModule(GenIntName) {}

class IntModuleTester extends Module {
  val intM1 = Module(new IntModuleTest)
  val intM2 = Module(new IntModuleParam("one", 1.0))
  val intM4 = Module(new IntModuleGenName("someIntName"))
}

class IntrinsicModuleSpec extends AnyFlatSpec with Matchers with FileCheck {
  (ChiselStage
    .emitCHIRRTL(new IntModuleTester)
    .split('\n')
    .map(_.trim.takeWhile(_ != '@'))
    should contain).allOf(
    "intmodule IntModuleTest : ",
    "intrinsic = TestIntrinsic",
    "intmodule IntModuleParam : ",
    "parameter STRING = \"one\"",
    "parameter REAL = 1.0",
    "intrinsic = OtherIntrinsic",
    "intmodule IntModuleGenName : ",
    "intrinsic = someIntName"
  )

  behavior.of("IntrinsicModule ports")

  they should "have source locator information" in {
    @nowarn("cat=deprecation")
    class IntModWithPort extends IntrinsicModule("TestIntrinsic") {
      val a = IO(Output(Bool()))
    }

    class Foo extends Module {
      val a = IO(Output(Bool()))

      private val intMod = Module(new IntModWithPort)
      a :<= intMod.a
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK: intmodule IntModWithPort :
           |CHECK:   output a : UInt<1> @[{{.+}}IntrinsicModule.scala {{[0-9]+}}:{{[0-9]+}}]
           |""".stripMargin
      )
  }
}
