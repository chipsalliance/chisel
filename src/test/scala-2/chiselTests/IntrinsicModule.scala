// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.experimental._
import chisel3.testers.BasicTester
import circt.stage.ChiselStage

class IntModuleTest extends IntrinsicModule("TestIntrinsic") {
  val foo = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

class IntModuleParam(str: String, dbl: Double)
    extends IntrinsicModule("OtherIntrinsic", Map("STRING" -> str, "REAL" -> dbl)) {}

class IntModuleGenName(GenIntName: String) extends IntrinsicModule(GenIntName) {}

class IntModuleTester extends BasicTester {
  val intM1 = Module(new IntModuleTest)
  val intM2 = Module(new IntModuleParam("one", 1.0))
  val intM4 = Module(new IntModuleGenName("someIntName"))
}

class IntrinsicModuleSpec extends ChiselFlatSpec {
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
}
