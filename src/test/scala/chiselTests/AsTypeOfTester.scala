// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.reflect.DataMirror
import chisel3.testers.BasicTester
import chisel3.experimental.Analog

class AsTypeOfBundleTester extends BasicTester {
  class MultiTypeBundle extends Bundle {
    val u = UInt(4.W)
    val s = SInt(4.W)
  }

  val bun = new MultiTypeBundle

  val bunAsTypeOf = ((4 << 4) + (15 << 0)).U.asTypeOf(bun)

  assert(bunAsTypeOf.u === 4.U)
  assert(bunAsTypeOf.s === -1.S)

  stop()
}

class AsTypeOfBundleZeroWidthTester extends BasicTester {
  class ZeroWidthBundle extends Bundle {
    val a = UInt(0.W)
    val b = UInt(1.W)
    val c = UInt(0.W)
  }

  val bun = new ZeroWidthBundle

  val bunAsTypeOf = 1.U.asTypeOf(bun)

  assert(bunAsTypeOf.a === 0.U)
  assert(bunAsTypeOf.b === 1.U)
  assert(bunAsTypeOf.c === 0.U)

  stop()
}

class AsTypeOfVecTester extends BasicTester {
  val vec = ((15 << 12) + (0 << 8) + (1 << 4) + (2 << 0)).U.asTypeOf(Vec(4, SInt(4.W)))

  assert(vec(0) === 2.S)
  assert(vec(1) === 1.S)
  assert(vec(2) === 0.S)
  assert(vec(3) === -1.S)

  stop()
}

class AsTypeOfTruncationTester extends BasicTester {
  val truncate = (64 + 3).U.asTypeOf(UInt(3.W))
  val expand = 1.U.asTypeOf(UInt(3.W))

  assert(DataMirror.widthOf(truncate).get == 3)
  assert(truncate === 3.U)
  assert(DataMirror.widthOf(expand).get == 3)
  assert(expand === 1.U)

  stop()
}

class ResetAsTypeOfBoolTester extends BasicTester {
  assert(reset.asTypeOf(Bool()) === reset.asBool)
  stop()
}

class AsTypeOfClockTester extends BasicTester {
  class MyBundle extends Bundle {
    val x = UInt(4.W)
    val y = Clock()
  }
  assert(true.B.asTypeOf(Clock()).asUInt.asBool === true.B)

  assert(0x1f.U.asTypeOf(new MyBundle).asUInt === 0x1f.U)
  stop()
}

class AsChiselEnumTester extends BasicTester {
  object MyEnum extends ChiselEnum {
    val foo, bar = Value
    val fizz = Value(2.U)
  }
  class MyBundle extends Bundle {
    val a = Bool()
    val b = Bool()
  }

  // To
  assert(2.U.asTypeOf(MyEnum()) === MyEnum.fizz)
  assert(VecInit(2.U.asBools).asTypeOf(MyEnum()) === MyEnum.fizz)
  assert(2.U.asTypeOf(new MyBundle).asTypeOf(MyEnum()) === MyEnum.fizz)

  // From
  assert(MyEnum.foo.asUInt === 0.U)
  val vec = MyEnum.bar.asTypeOf(Vec(2, Bool()))
  assert(vec(0) === 1.U)
  assert(vec(1) === 0.U)
  val bun = MyEnum.fizz.asTypeOf(new MyBundle)
  assert(bun.b === 0.U)
  assert(bun.a === 1.U)

  // In aggregate
  class OtherBundle extends Bundle {
    val myEnum = MyEnum()
    val foo = Bool()
  }
  val wire = Wire(new OtherBundle)
  wire.myEnum := MyEnum.fizz
  wire.foo := true.B

  assert(wire.asUInt === 5.U)
  val other = 5.U.asTypeOf(new OtherBundle)
  assert(other.myEnum === MyEnum.fizz)
  assert(other.foo === true.B)

  stop()
}

object AsTypeOfSpec {
  class MyBundle(w: Int) extends Bundle {
    val a = UInt(w.W)
    val b = UInt(w.W)
  }
  class MyBundleUnknownWidth(w: Int) extends Bundle {
    val a = UInt(w.W)
    val b = UInt()
  }
  class OtherBundle extends Bundle {
    val foo = UInt(8.W)
    val bar = UInt(8.W)
  }
  object MyEnum extends ChiselEnum {
    val a, b = Value
    val c = Value(5.U)
  }

  class SimpleAsTypeOf[A <: Data, B <: Data](gen1: A, gen2: B) extends RawModule {
    val in = IO(Input(gen1))
    val out = IO(Output(gen2))
    out :#= in.asTypeOf(out)
  }
  class WireTrampolineAsTypeOf[A <: Data, B <: Data](gen1: A, gen2: B)(driveWire: A => Unit) extends RawModule {
    val w = Wire(gen1)
    dontTouch(w)
    val out = IO(Output(gen2))
    driveWire(w)
    out := w.asTypeOf(out)
  }
}

class AsTypeOfSpec extends ChiselFunSpec {
  import AsTypeOfSpec._

  describe("asTypeOf") {

    it("should work with Bundles containing Bits Types") {
      assertTesterPasses { new AsTypeOfBundleTester }
    }

    it("should work with Bundles that have fields of zero width") {
      assertTesterPasses { new AsTypeOfBundleZeroWidthTester }
    }

    it("should work with Vecs containing Bits Types") {
      assertTesterPasses { new AsTypeOfVecTester }
    }

    it("should expand and truncate UInts of different width") {
      assertTesterPasses { new AsTypeOfTruncationTester }
    }

    it("should work for casting implicit Reset to Bool") {
      assertTesterPasses { new ResetAsTypeOfBoolTester }
    }

    it("should work for casting to and from ChiselEnums") {
      assertTesterPasses(new AsChiselEnumTester)
    }

    it("should work for casting to and from Clock") {
      assertTesterPasses(new AsTypeOfClockTester)
    }
  }

  describe("asTypeOf Bundle") {

    it("should error if the target Bundle type has unknown width") {
      val sourceTypes = Seq(
        new MyBundle(8),
        Vec(2, UInt(8.W)),
        UInt(16.W),
        SInt(16.W),
        Bool(),
        Clock(),
        Reset(),
        Analog(8.W),
        AsyncReset()
      )
      for (sourceType <- sourceTypes) {
        val e = the[ChiselException] thrownBy {
          ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle(8), new MyBundleUnknownWidth(8)))
        }
        (e.getMessage should include).regex("Width of.*is unknown")
      }
    }

    describe("from Bundles") {
      they("should expand from narrower to wider Bundle") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle(4), new MyBundle(8)))
        verilog should include("assign out_a = 8'h0;")
        verilog should include("assign out_b = {in_a, in_b};")
      }
      they("should truncate from wider to narrower Bundle") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle(8), new MyBundle(4)))
        verilog should include("assign out_a = in_b[7:4];")
        verilog should include("assign out_b = in_b[3:0];")
      }
      they("should match for Bundles of same width") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle(8), new OtherBundle))
        verilog should include("assign out_bar = in_b;")
        verilog should include("assign out_foo = in_a;")
      }
      they("should work if the source Bundle has unknown width") {
        val verilog =
          ChiselStage.emitSystemVerilog(new WireTrampolineAsTypeOf(new MyBundleUnknownWidth(8), new MyBundle(8))({ w =>
            w.a := 0.U(8.W)
            w.b := 0.U(8.W)
          }))
        verilog should include("assign out_a = w_a;")
        verilog should include("assign out_b = w_b;")
      }
    }
    describe("from Vecs") {
      they("should expand from narrower Vec to wider Bundle") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Vec(2, UInt(4.W)), new MyBundle(8)))
        verilog should include("assign out_a = 8'h0;")
        verilog should include("assign out_b = {in_1, in_0};")
      }
      they("should truncate from wider Vecto narrower Bundle") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Vec(2, UInt(8.W)), new MyBundle(4)))
        verilog should include("assign out_a = in_0[7:4];")
        verilog should include("assign out_b = in_0[3:0];")
      }
      they("should match for Vecs of same width") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Vec(2, UInt(8.W)), new MyBundle(8)))
        verilog should include("assign out_a = in_1;")
        verilog should include("assign out_b = in_0;")
      }
      they("should work if the source Vec has unknown width") {
        val verilog =
          ChiselStage.emitSystemVerilog(new WireTrampolineAsTypeOf(Vec(2, UInt()), new MyBundle(8))({ w =>
            w(0) := 0.U(8.W)
            w(1) := 0.U(8.W)
          }))
        verilog should include("assign out_a = w_1;")
        verilog should include("assign out_b = w_0;")
      }
    }
    describe("from UInts and SInts") {
      val mkType: Seq[(String, Width => Bits)] = Seq("UInt" -> UInt.apply, "SInt" -> SInt.apply)
      for ((name, gen) <- mkType) {
        they(s"should expand from narrower $name to wider Bundle") {
          val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(gen(4.W), new MyBundle(8)))
          verilog should include("assign out_a = 8'h0;")
          verilog should include("assign out_b = {4'h0, in};")
        }
        they(s"should truncate from wider $name to narrower Bundle") {
          val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(gen(16.W), new MyBundle(4)))
          verilog should include("assign out_a = in[7:4];")
          verilog should include("assign out_b = in[3:0];")
        }
        they(s"should match for $name of same width") {
          val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(gen(16.W), new MyBundle(8)))
          verilog should include("assign out_a = in[15:8];")
          verilog should include("assign out_b = in[7:0];")
        }
      }
      they("should work if the source UInt has unknown width") {
        val verilog =
          ChiselStage.emitSystemVerilog(new WireTrampolineAsTypeOf(UInt(), new MyBundle(8))({ w =>
            w := 123.U(16.W)
          }))
        verilog should include("assign out_a = w[15:8];")
        verilog should include("assign out_b = w[7:0];")
      }
      they("should work if the source SInt has unknown width") {
        val verilog =
          ChiselStage.emitSystemVerilog(new WireTrampolineAsTypeOf(SInt(), new MyBundle(8))({ w =>
            w := -123.S(16.W)
          }))
        verilog should include("assign out_a = w[15:8];")
        verilog should include("assign out_b = w[7:0];")
      }
    }
    val tpes = List(Bool(), Clock(), AsyncReset())
    for (tpe <- tpes) {
      they(s"should expand from $tpe to wider Bundle") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(tpe, new MyBundle(8)))
        verilog should include("assign out_a = 8'h0;")
        verilog should include("assign out_b = {7'h0, in};")
      }
    }
    they("should truncate from wider Reset to narrower Bundle") {
      val verilog = ChiselStage.emitSystemVerilog(new WireTrampolineAsTypeOf(Reset(), new MyBundle(8))({ w =>
        w := false.B
      }))
      verilog should include("assign out_a = 8'h0;")
      verilog should include("assign out_b = {7'h0, w};")
    }
    describe("from ChiselEnums") {
      they("should expand from narrower ChiselEnum to wider Bundle") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(MyEnum(), new MyBundle(8)))
        verilog should include("assign out_a = 8'h0;")
        verilog should include("assign out_b = {5'h0, in};")
      }
    }
  }
}
