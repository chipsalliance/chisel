// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.experimental._
import chisel3.reflect.DataMirror
import chisel3.testers.{BasicTester, TesterDriver}
import circt.stage.ChiselStage

class IntModuleTest extends IntModule("TestIntrinsic") {
  val foo = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

class IntModuleStringParam(str: String) extends IntModule("OtherIntrinsic", Map("STRING" -> str)) {
  val io = IO(new Bundle {
    val out = UInt(32.W)
  })
}

class IntModuleRealParam(dbl: Double) extends IntModule("OtherIntrinsic", Map("REAL" -> dbl)) {
  val io = IO(new Bundle {
    val out = UInt(64.W)
  })
}

class IntModuleNoIO(GenIntName: String) extends IntModule(GenIntName) {
  // Whoops! typo
  val ioo = IO(new Bundle {
    val out = Output(UInt(8.W))
  })
}

class IntModuleTester extends BasicTester {
  val intM1 = Module(new IntModuleTest)
  val intM2 = Module(new IntModuleStringParam("one"))
  val intM3 = Module(new IntModuleRealParam(1.0))
  val intM4 = Module(new IntModuleNoIO("someIntName"))
}

class IntModuleSpec extends ChiselFlatSpec {
  def myGenerateFirrtl(t: => Module): String = ChiselStage.emitCHIRRTL(t)
  val firrtlOutput = myGenerateFirrtl(new IntModuleTester)
  firrtlOutput should include("intmodule IntModuleTest")
  firrtlOutput should include("intrinsic = TestIntrinsic")
  firrtlOutput should include("intmodule IntModuleStringParam")
  firrtlOutput should include("parameter STRING = \"one\"")
  firrtlOutput should include("intmodule IntModuleRealParam")
  firrtlOutput should include("intrinsic = OtherIntrinsic")
  firrtlOutput should include("intmodule IntModuleNoIO")
  firrtlOutput should include("intrinsic = someIntName")
}
