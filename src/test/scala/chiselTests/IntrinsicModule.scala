// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.experimental._
import chisel3.reflect.DataMirror
import chisel3.testers.{BasicTester, TesterDriver}
import circt.stage.ChiselStage

class IntModuleTest extends IntrinsicModule("TestIntrinsic") {
  val foo = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

class IntModuleStringParam(str: String) extends IntrinsicModule("OtherIntrinsic", Map("STRING" -> str)) {}

class IntModuleRealParam(dbl: Double) extends IntrinsicModule("OtherIntrinsic", Map("REAL" -> dbl)) {}

class IntModuleGenName(GenIntName: String) extends IntrinsicModule(GenIntName) {}

class IntModuleTester extends BasicTester {
  val intM1 = Module(new IntModuleTest)
  val intM2 = Module(new IntModuleStringParam("one"))
  val intM3 = Module(new IntModuleRealParam(1.0))
  val intM4 = Module(new IntModuleGenName("someIntName"))
}

class IntrinsicModuleSpec extends ChiselFlatSpec {
  (ChiselStage
    .emitCHIRRTL(new IntModuleTester)
    .split('\n')
    .map(x => x.trim)
    should contain).allOf(
    "intmodule IntModuleTest :",
    "intrinsic = TestIntrinsic",
    "intmodule IntModuleStringParam :",
    "parameter STRING = \"one\"",
    "intmodule IntModuleRealParam :",
    "intrinsic = OtherIntrinsic",
    "intmodule IntModuleGenName :",
    "intrinsic = someIntName"
  )
}
