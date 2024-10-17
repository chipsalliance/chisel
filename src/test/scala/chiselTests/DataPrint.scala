// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.BundleLiterals._
import circt.stage.ChiselStage
import org.scalatest.matchers.should.Matchers

class DataPrintSpec extends ChiselFlatSpec with Matchers {
  object EnumTest extends ChiselEnum {
    val sNone, sOne, sTwo = Value
  }

  class BundleTest extends Bundle {
    val a = UInt(8.W)
    val b = Bool()
  }

  class PartialBundleTest extends Bundle {
    val a = UInt(8.W)
    val b = Bool()
    val c = SInt(8.W)
    val f = EnumTest.Type()
  }

  object A extends layer.Layer(layer.LayerConfig.Extract()) {
    object B extends layer.Layer(layer.LayerConfig.Extract())
  }

  "Data types" should "have a meaningful string representation" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        UInt().toString should be("UInt")
        UInt(8.W).toString should be("UInt<8>")
        probe.Probe(UInt(8.W)).toString should be("Probe<UInt<8>>")
        probe.RWProbe(UInt(8.W)).toString should be("RWProbe<UInt<8>>")
        probe.Probe(UInt(8.W), A).toString should be("Probe[A]<UInt<8>>")
        probe.Probe(UInt(8.W), A.B).toString should be("Probe[A.B]<UInt<8>>")
        SInt(15.W).toString should be("SInt<15>")
        Bool().toString should be("Bool")
        Clock().toString should be("Clock")
        Vec(3, UInt(2.W)).toString should be("UInt<2>[3]")
        EnumTest.Type().toString should be("EnumTest")
        (new BundleTest).toString should be("BundleTest")
        (probe.Probe(new BundleTest)).toString should be("Probe<BundleTest>")
        new Bundle { val a = UInt(8.W) }.toString should be("AnonymousBundle")
        new Bundle { val a = UInt(8.W) }.a.toString should be("UInt<8>")
      }
    }
  }

  class BoundDataModule extends Module { // not in the test to avoid anon naming suffixes
    Wire(UInt()).toString should be("BoundDataModule.?: Wire[UInt]")
    Wire(probe.Probe(UInt(1.W))).toString should be("BoundDataModule.?: Wire[Probe<UInt<1>>]")
    Wire(probe.Probe(UInt(1.W), A)).toString should be("BoundDataModule.?: Wire[Probe[A]<UInt<1>>]")
    Reg(SInt()).toString should be("BoundDataModule.?: Reg[SInt]")
    val io = IO(Output(Bool())) // needs a name so elaboration doesn't fail
    io.toString should be("BoundDataModule.io: IO[Bool]")
    val m = Mem(4, UInt(2.W))
    m(2).toString should be("BoundDataModule.?: MemPort[UInt<2>]")
    (2.U + 2.U).toString should be("BoundDataModule.?: OpResult[UInt<2>]")
    Wire(Vec(3, UInt(2.W))).toString should be("BoundDataModule.?: Wire[UInt<2>[3]]")

    val idx = IO(Input(UInt(3.W)))
    val jdx = IO(Input(new Bundle {
      val value = UInt(3.W)
    }))
    val port = IO(Input(new Bundle {
      val vec = Vec(8, UInt(8.W))
    }))
    port.vec(idx).toString should be("BoundDataModule.port.vec[idx]: IO[UInt<8>]")
    port.vec(jdx.value).toString should be("BoundDataModule.port.vec[jdx.value]: IO[UInt<8>]")
    port.vec(3.U).toString should be("BoundDataModule.port.vec[3]: IO[UInt<8>]")

    class InnerModule extends Module {
      val io = IO(Output(new Bundle {
        val a = UInt(4.W)
      }))
    }
    val inner = Module(new InnerModule)
    inner.clock.toString should be("InnerModule.clock: IO[Clock]")
    inner.io.a.toString should be("InnerModule.io.a: IO[UInt<4>]")

    class FooTypeTest extends Bundle {
      val foo = Vec(2, UInt(8.W))
      val fizz = UInt(8.W)
    }
    val tpe = new FooTypeTest
    val fooio: FooTypeTest = IO(Input(tpe))
    fooio.foo(0).toString should be("BoundDataModule.fooio.foo[0]: IO[UInt<8>]")

    class NestedBundle extends Bundle {
      val nestedFoo = UInt(8.W)
      val nestedFooVec = Vec(2, UInt(8.W))
    }
    class NestedType extends Bundle {
      val foo = new NestedBundle
    }

    val nestedTpe = new NestedType
    val nestedio = IO(Input(nestedTpe))
    (nestedio.foo.nestedFoo.toString should be("BoundDataModule.nestedio.foo.nestedFoo: IO[UInt<8>]"))
    (nestedio.foo.nestedFooVec(0).toString should be("BoundDataModule.nestedio.foo.nestedFooVec[0]: IO[UInt<8>]"))
  }

  "Bound data types" should "have a meaningful string representation" in {
    ChiselStage.emitCHIRRTL { new BoundDataModule }
  }

  "Literals" should "have a meaningful string representation" in {
    3.U.toString should be("UInt<2>(3)")
    3.U(5.W).toString should be("UInt<5>(3)")
    -1.S.toString should be("SInt<1>(-1)")
    false.B.toString should be("Bool(false)")
    true.B.toString should be("Bool(true)")
    Vec(3, UInt(4.W)).toString should be("UInt<4>[3]")
    EnumTest.sNone.toString should be("EnumTest(0=sNone)")
    EnumTest.sTwo.toString should be("EnumTest(2=sTwo)")
    EnumTest(1.U).toString should be("EnumTest(1=sOne)")
    (new BundleTest).Lit(_.a -> 2.U, _.b -> false.B).toString should be("BundleTest(a=UInt<8>(2), b=Bool(false))")
    (new PartialBundleTest).Lit().toString should be(
      "PartialBundleTest(a=UInt<8>(DontCare), b=Bool(DontCare), c=SInt<8>(DontCare), f=EnumTest(DontCare))"
    )
    DontCare.toString should be("DontCare()")
  }
}
