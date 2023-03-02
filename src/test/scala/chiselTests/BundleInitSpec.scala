// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.BundleLiterals.AddBundleLiteralConstructor
import chisel3.testers.BasicTester
import chisel3.util.Counter
import circt.stage.ChiselStage
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals.AddVecLiteralConstructor
import chisel3.experimental.BundleLiteralException

import scala.language.reflectiveCalls

class BundleInitSpec extends ChiselFreeSpec with Utils {
  object MyEnum extends ChiselEnum {
    val sA, sB = Value
  }
  object MyEnumB extends ChiselEnum {
    val sA, sB = Value
  }
  class MyBundle extends Bundle {
    val a = UInt(8.W)
    val b = Bool()
    val c = MyEnum()
  }

  class LongBundle extends Bundle {
    val a = UInt(48.W)
    val b = SInt(32.W)
    val c = UInt(16.W)
  }
  
  //NOTE: I had problems where this would not work if this class declaration was inside test scope
  class HasBundleInit extends Module {
    val initValue =(new MyBundle).init(_.a -> 0xab.U(8.W), _.b -> false.B, _.c -> MyEnum.sB)
    val y = RegInit(initValue)
  }

  "Bundle Init should work when used to initialize all elements in Bundle" in {
    val firrtl = ChiselStage.emitCHIRRTL(new HasBundleInit)
    firrtl should include("""_y_WIRE[0] <= UInt<8>("hab")""")
    firrtl should include("""_y_WIRE[1] <= UInt<1>("h0")""")
    y.c.toString should include(MyEnum.sB.toString)
    firrtl should include("""      reset => (reset, _y_WIRE)""".stripMargin)
  }

}
