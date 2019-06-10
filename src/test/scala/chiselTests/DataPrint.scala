// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.{ChiselEnum, FixedPoint, RawModule, MultiIOModule}
import chisel3.experimental.BundleLiterals._

class DataPrintSpec extends ChiselFlatSpec with Matchers {
  object EnumTest extends ChiselEnum {
    val sNone, sOne, sTwo = Value
  }

  class BundleTest extends Bundle {
    val a = UInt(8.W)
    val b = Bool()
  }

  "Data types" should "have a meaningful string representation" in {
    elaborate { new RawModule {
      UInt().toString should be ("UInt")
      UInt(8.W).toString should be ("UInt<8>")
      SInt(15.W).toString should be ("SInt<15>")
      Bool().toString should be ("Bool")
      Clock().toString should be ("Clock")
      FixedPoint(5.W, 3.BP).toString should be ("FixedPoint<5><<3>>")
      Vec(3, UInt(2.W)).toString should be ("UInt<2>[3]")
      EnumTest.Type().toString should be ("EnumTest")
      (new BundleTest).toString should be ("BundleTest")
    } }
  }

  class BoundDataModule extends MultiIOModule {  // not in the test to avoid anon naming suffixes
    Wire(UInt()).toString should be("UInt(Wire in DataPrintSpec$BoundDataModule)")
    Reg(SInt()).toString should be("SInt(Reg in DataPrintSpec$BoundDataModule)")
    val io = IO(Output(Bool()))  // needs a name so elaboration doesn't fail
    io.toString should be("Bool(IO in unelaborated DataPrintSpec$BoundDataModule)")
    val m = Mem(4, UInt(2.W))
    m(2).toString should be("UInt<2>(MemPort in DataPrintSpec$BoundDataModule)")
    (2.U + 2.U).toString should be("UInt<2>(OpResult in DataPrintSpec$BoundDataModule)")
    Wire(Vec(3, UInt(2.W))).toString should be ("UInt<2>[3](Wire in DataPrintSpec$BoundDataModule)")

    class InnerModule extends MultiIOModule {
      val io = IO(Output(new Bundle {
        val a = UInt(4.W)
      }))
    }
    val inner = Module(new InnerModule)
    inner.clock.toString should be ("Clock(IO clock in DataPrintSpec$BoundDataModule$InnerModule)")
    inner.io.a.toString should be ("UInt<4>(IO io_a in DataPrintSpec$BoundDataModule$InnerModule)")
  }

  "Bound data types" should "have a meaningful string representation" in {
    elaborate { new BoundDataModule }
  }

  "Literals" should "have a meaningful string representation" in {
    elaborate { new RawModule {
      3.U.toString should be ("UInt<2>(3)")
      3.U(5.W).toString should be ("UInt<5>(3)")
      -1.S.toString should be ("SInt<1>(-1)")
      false.B.toString should be ("Bool(false)")
      true.B.toString should be ("Bool(true)")
      2.25.F(6.W, 2.BP).toString should be ("FixedPoint<6><<2>>(2.25)")
      -2.25.F(6.W, 2.BP).toString should be ("FixedPoint<6><<2>>(-2.25)")
      EnumTest.sNone.toString should be ("EnumTest(0=sNone)")
      EnumTest.sTwo.toString should be ("EnumTest(2=sTwo)")
      EnumTest(1.U).toString should be ("EnumTest(1=sOne)")
      (new BundleTest).Lit(_.a -> 2.U, _.b -> false.B).toString should be ("BundleTest(a=UInt<8>(2), b=Bool(false))")
      new Bundle {
        val a = UInt(8.W)
      }.toString should be ("AnonymousBundle")
      DontCare.toString should be ("DontCare()")
    } }
  }
}
