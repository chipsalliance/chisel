// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.reflect.DataMirror
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.experimental.Analog
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class AsTypeOfBundleTester extends Module {
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

class AsTypeOfBundleZeroWidthTester extends Module {
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

class AsTypeOfVecTester extends Module {
  val vec = ((15 << 12) + (0 << 8) + (1 << 4) + (2 << 0)).U.asTypeOf(Vec(4, SInt(4.W)))

  assert(vec(0) === 2.S)
  assert(vec(1) === 1.S)
  assert(vec(2) === 0.S)
  assert(vec(3) === -1.S)

  stop()
}

class AsTypeOfTruncationTester extends Module {
  val truncate = (64 + 3).U.asTypeOf(UInt(3.W))
  val expand = 1.U.asTypeOf(UInt(3.W))

  assert(DataMirror.widthOf(truncate).get == 3)
  assert(truncate === 3.U)
  assert(DataMirror.widthOf(expand).get == 3)
  assert(expand === 1.U)

  stop()
}

class ResetAsTypeOfBoolTester extends Module {
  assert(reset.asTypeOf(Bool()) === reset.asBool)
  stop()
}

class AsTypeOfClockTester extends Module {
  class MyBundle extends Bundle {
    val x = UInt(4.W)
    val y = Clock()
  }
  assert(true.B.asTypeOf(Clock()).asUInt.asBool === true.B)

  assert(0x1f.U.asTypeOf(new MyBundle).asUInt === 0x1f.U)
  stop()
}

class AsChiselEnumTester extends Module {
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
  class MyBundle extends Bundle {
    val a = UInt(4.W)
    val b = UInt(4.W)
  }
  class MyBundleUnknownWidth(w: Int) extends Bundle {
    val a = UInt(w.W)
    val b = UInt()
  }
  object MyEnum extends ChiselEnum {
    val a, b = Value
    val c = Value(10.U)
  }

  class SimpleAsTypeOf[A <: Data, B <: Data](gen1: A, gen2: B) extends RawModule {
    val in = IO(Input(gen1))
    // Out is a wire so we can use inferred widths
    val out = dontTouch(Wire(gen2))
    out :#= in.asTypeOf(out)
  }
  class WireSourceAsTypeOf[A <: Data, B <: Data](gen1: A, gen2: B)(driveIn: A => Unit) extends RawModule {
    // Called in so that tests can check for same Verilog from this and SimpleAsTypeOf
    val in = dontTouch(Wire(gen1))
    driveIn(in)
    val out = dontTouch(Wire(gen2))
    out := in.asTypeOf(out)
  }
}

class AsTypeOfSpec extends AnyFunSpec with Matchers with ChiselSim {
  import AsTypeOfSpec._

  describe("asTypeOf") {

    it("should work with Bundles containing Bits Types") {
      simulate { new AsTypeOfBundleTester }(RunUntilFinished(3))
    }

    it("should work with Bundles that have fields of zero width") {
      simulate { new AsTypeOfBundleZeroWidthTester }(RunUntilFinished(3))
    }

    it("should work with Vecs containing Bits Types") {
      simulate { new AsTypeOfVecTester }(RunUntilFinished(3))
    }

    it("should expand and truncate UInts of different width") {
      simulate { new AsTypeOfTruncationTester }(RunUntilFinished(3))
    }

    it("should work for casting implicit Reset to Bool") {
      simulate { new ResetAsTypeOfBoolTester }(RunUntilFinished(3))
    }

    it("should work for casting to and from ChiselEnums") {
      simulate(new AsChiselEnumTester)(RunUntilFinished(3))
    }

    it("should work for casting to and from Clock") {
      simulate(new AsTypeOfClockTester)(RunUntilFinished(3))
    }
  }

  describe("Bundles") {
    describe("as the target type") {
      it("should error if the Bundle has unknown width") {
        val e = the[ChiselException] thrownBy {
          ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(16.W), new MyBundleUnknownWidth(8)))
        }
        (e.getMessage should include).regex("Width of.*is unknown")
      }
      they("should expand narrower inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(6.W), new MyBundle),
          () => new WireSourceAsTypeOf(UInt(), new MyBundle)(_ := 0.U(6.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [3:0] out_a = {2'h0, in[5:4]};")
          verilog should include("wire [3:0] out_b = in[3:0];")
        }
      }
      they("should truncate wider inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(10.W), new MyBundle),
          () => new WireSourceAsTypeOf(UInt(), new MyBundle)(_ := 0.U(10.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [3:0] out_a = in[7:4];")
          verilog should include("wire [3:0] out_b = in[3:0];")
        }
      }
      they("should match for matched width inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(8.W), new MyBundle),
          () => new WireSourceAsTypeOf(UInt(), new MyBundle)(_ := 0.U(8.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [3:0] out_a = in[7:4];")
          verilog should include("wire [3:0] out_b = in[3:0];")
        }
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle, UInt(10.W)))
        verilog should include("wire [9:0] out = {2'h0, in_a, in_b};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle, UInt(6.W)))
        verilog should include("wire [5:0] out = {in_a[1:0], in_b};")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle, UInt(8.W)))
        verilog should include("wire [7:0] out = {in_a, in_b};")
      }
      they("should work for inferred width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(new MyBundle, UInt()))
        verilog should include("wire [7:0] out = {in_a, in_b};")
      }
      they("should work if the source Bundle has unknown width") {
        val verilog =
          ChiselStage.emitSystemVerilog(new WireSourceAsTypeOf(new MyBundleUnknownWidth(4), UInt(8.W))({ in =>
            in.a := 0.U(4.W)
            in.b := 0.U(4.W)
          }))
        verilog should include("wire [7:0] out = {in_a, in_b};")
      }
    }
  }

  describe("Vecs") {
    describe("as the target type") {
      it("should error if the Vec has unknown width") {
        val e = the[ChiselException] thrownBy {
          ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(16.W), Vec(2, UInt())))
        }
        (e.getMessage should include).regex("Width of.*is unknown")
      }
      they("should expand narrower inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(6.W), Vec(2, UInt(4.W))),
          () => new WireSourceAsTypeOf(UInt(), Vec(2, UInt(4.W)))(_ := 0.U(6.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [3:0] out_0 = in[3:0];")
          verilog should include("wire [3:0] out_1 = {2'h0, in[5:4]};")
        }
      }
      they("should truncate wider inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(10.W), Vec(2, UInt(4.W))),
          () => new WireSourceAsTypeOf(UInt(), Vec(2, UInt(4.W)))(_ := 0.U(10.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [3:0] out_0 = in[3:0];")
          verilog should include("wire [3:0] out_1 = in[7:4];")
        }
      }
      they("should match for matched width inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(8.W), Vec(2, UInt(4.W))),
          () => new WireSourceAsTypeOf(UInt(), Vec(2, UInt(4.W)))(_ := 0.U(8.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [3:0] out_0 = in[3:0];")
          verilog should include("wire [3:0] out_1 = in[7:4];")
        }
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Vec(2, UInt(4.W)), UInt(10.W)))
        verilog should include("wire [9:0] out = {2'h0, in_1, in_0};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Vec(2, UInt(4.W)), UInt(6.W)))
        verilog should include("wire [5:0] out = {in_1[1:0], in_0};")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Vec(2, UInt(4.W)), UInt(8.W)))
        verilog should include("wire [7:0] out = {in_1, in_0};")
      }
      they("should work for inferred width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Vec(2, UInt(4.W)), UInt()))
        verilog should include("wire [7:0] out = {in_1, in_0};")
      }
      they("should work if the source Vec has unknown width") {
        val verilog =
          ChiselStage.emitSystemVerilog(new WireSourceAsTypeOf(Vec(2, UInt()), UInt(8.W))({ w =>
            w(0) := 0.U(4.W)
            w(1) := 0.U(4.W)
          }))
        verilog should include("wire [7:0] out = {in_1, in_0};")
      }
    }
  }

  describe("ChiselEnums") {
    describe("as the target type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(2.W), MyEnum()))
        verilog should include("wire [3:0] out = {2'h0, in};")
      }
      they("should error on wider inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), MyEnum()))
        e.getMessage should include(
          "The UInt being cast to chiselTests.AsTypeOfSpec$MyEnum is wider than chiselTests.AsTypeOfSpec$MyEnum's width (4)"
        )
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(4.W), MyEnum()))
        verilog should include("wire [3:0] out = in;")
      }
      they("should error on unknown width inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(), MyEnum()))
        e.getMessage should include(
          "Non-literal UInts being cast to chiselTests.AsTypeOfSpec$MyEnum must have a defined width"
        )
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(MyEnum(), UInt(8.W)))
        verilog should include("wire [7:0] out = {4'h0, in};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(MyEnum(), UInt(3.W)))
        verilog should include("wire [2:0] out = in[2:0];")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(MyEnum(), UInt(4.W)))
        verilog should include("wire [3:0] out = in;")
      }
      they("should work for unknown width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(MyEnum(), UInt()))
        verilog should include("wire [3:0] out = in;")
      }
    }
  }

  describe("SInts") {
    describe("as the target type") {
      they("should expand narrower inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(6.W), SInt(8.W)),
          () => new WireSourceAsTypeOf(UInt(), SInt(8.W))(_ := 0.U(6.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [7:0] out = {{2{in[5]}}, in};")
        }
      }
      they("should truncate wider inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(10.W), SInt(8.W)),
          () => new WireSourceAsTypeOf(UInt(), SInt(8.W))(_ := 0.U(10.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [7:0] out = in[7:0];")
        }
      }
      they("should match for matched width inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(8.W), SInt(8.W)),
          () => new WireSourceAsTypeOf(UInt(), SInt(8.W))(_ := 0.U(8.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [7:0] out = in;")
        }
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(SInt(8.W), UInt(10.W)))
        verilog should include("wire [9:0] out = {2'h0, in};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(SInt(8.W), UInt(6.W)))
        verilog should include("wire [5:0] out = in[5:0];")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(SInt(8.W), UInt(8.W)))
        verilog should include("wire [7:0] out = in;")
      }
      they("should work for unknown width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(SInt(8.W), UInt()))
        verilog should include("wire [7:0] out = in;")
      }
    }
  }

  describe("UInts") {
    describe("as the target type") {
      they("should expand narrower inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(6.W), UInt(8.W)),
          () => new WireSourceAsTypeOf(UInt(), UInt(8.W))(_ := 0.U(6.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [7:0] out = {2'h0, in};")
        }
      }
      they("should truncate wider inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(10.W), UInt(8.W)),
          () => new WireSourceAsTypeOf(UInt(), UInt(8.W))(_ := 0.U(10.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [7:0] out = in[7:0];")
        }
      }
      they("should match for matched width inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(8.W), UInt(8.W)),
          () => new WireSourceAsTypeOf(UInt(), UInt(8.W))(_ := 0.U(8.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire [7:0] out = in;")
        }
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), UInt(10.W)))
        verilog should include("wire [9:0] out = {2'h0, in};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), UInt(6.W)))
        verilog should include("wire [5:0] out = in[5:0];")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), UInt(8.W)))
        verilog should include("wire [7:0] out = in;")
      }
      they("should work for inferred width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), UInt()))
        verilog should include("wire [7:0] out = in;")
      }
    }
  }

  describe("Analogs") {
    describe("as the target type") {
      they("should error") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(
          new RawModule {
            val in = IO(Input(UInt(8.W)))
            val out = IO(Analog(8.W))
            out := in.asTypeOf(out)
          },
          Array("--throw-on-first-error")
        )
        e.getMessage should include("Analog does not support fromUInt")
      }
    }
    describe("as the source type") {
      they("should error") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new RawModule {
          val in = IO(Analog(8.W))
          val out = IO(Output(UInt(8.W)))
          out := in.asTypeOf(out)
        })
        e.getMessage should include("Analog does not support asUInt")
      }
    }
  }

  describe("Bools") {
    describe("as the target type") {
      they("should expand narrower inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(0.W), Bool()),
          () => new WireSourceAsTypeOf(UInt(), Bool())(_ := 0.U(0.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire out = 1'h0;")
        }
      }
      they("should truncate wider inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(8.W), Bool()),
          () => new WireSourceAsTypeOf(UInt(), Bool())(_ := 0.U(8.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          // 8-bit wire for inferred test puts several spaces between wire and out
          (verilog should include).regex("""wire +out = in\[0\];""")
        }
      }
      they("should match for matched width inputs") {
        val tests = List(
          () => new SimpleAsTypeOf(UInt(1.W), Bool()),
          () => new WireSourceAsTypeOf(UInt(), Bool())(_ := 0.U(1.W))
        )
        for (test <- tests) {
          val verilog = ChiselStage.emitSystemVerilog(test())
          verilog should include("wire out = in;")
        }
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Bool(), UInt(8.W)))
        verilog should include("wire [7:0] out = {7'h0, in};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Bool(), UInt(0.W)))
        verilog shouldNot include("assign")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Bool(), UInt(1.W)))
        verilog should include("wire out = in;")
      }
      they("should work for inferred width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Bool(), UInt()))
        verilog should include("wire out = in;")
      }
    }
  }

  describe("AsyncReset") {
    describe("as the target type") {
      they("should error on narrower inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(0.W), AsyncReset()))
        e.getMessage should include("can't covert UInt<0> to Bool")
      }
      they("should error on wider inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), AsyncReset()))
        e.getMessage should include("can't covert UInt<8> to Bool")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(1.W), AsyncReset()))
        verilog should include("wire out = in;")
      }
      they("should error on inferred width inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(), AsyncReset()))
        e.getMessage should include("can't covert UInt to Bool")
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(AsyncReset(), UInt(8.W)))
        verilog should include("wire [7:0] out = {7'h0, in};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(AsyncReset(), UInt(0.W)))
        verilog shouldNot include("assign")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(AsyncReset(), UInt(1.W)))
        verilog should include("wire out = in;")
      }
      they("should work for inferred width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(AsyncReset(), UInt()))
        verilog should include("wire out = in;")
      }
    }
  }

  describe("Clock") {
    describe("as the target type") {
      they("should error on narrower inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(0.W), Clock()))
        e.getMessage should include("can't covert UInt<0> to Bool")
      }
      they("should error on wider inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), Clock()))
        e.getMessage should include("can't covert UInt<8> to Bool")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(1.W), Clock()))
        verilog should include("wire out = in;")
      }
      they("should error on inferred width inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(), Clock()))
        e.getMessage should include("can't covert UInt to Bool")
      }
    }
    describe("as the source type") {
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Clock(), UInt(8.W)))
        verilog should include("wire [7:0] out = {7'h0, in};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Clock(), UInt(0.W)))
        verilog shouldNot include("assign")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Clock(), UInt(1.W)))
        verilog should include("wire out = in;")
      }
      they("should work for inferred width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(Clock(), UInt()))
        verilog should include("wire out = in;")
      }
    }
  }

  describe("Reset") {
    describe("as the target type") {
      // It feels a little weird that these errors are inconsistent with those of Clock and AsyncReset,
      // but reset inference works via how abstract Reset is driven, so it can't be driven with other types
      // We could possibly make it work with Bool, AsyncReset, and Reset as source types, but users should
      // just elide the .asTypeOf in those cases
      they("should error on narrower inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(0.W), Reset()))
        e.getMessage should include("Sink (Reset) and Source (UInt<0>) have different types")
      }
      they("should error on wider inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(8.W), Reset()))
        e.getMessage should include("Sink (Reset) and Source (UInt<8>) have different types")
      }
      they("should error on for matched width inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(1.W), Reset()))
        e.getMessage should include("Sink (Reset) and Source (UInt<1>) have different types")
      }
      they("should error on inferred width inputs") {
        val e = the[ChiselException] thrownBy ChiselStage.emitSystemVerilog(new SimpleAsTypeOf(UInt(), Reset()))
        e.getMessage should include("Sink (Reset) and Source (UInt) have different types")
      }
    }
    describe("as the source type") {
      // Have to use trampoline wire because inferred reset cannot be top-level ports
      they("should expand narrower inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new WireSourceAsTypeOf(Reset(), UInt(8.W))(_ := false.B))
        verilog should include("wire [7:0] out = {7'h0, in};")
      }
      they("should truncate wider inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new WireSourceAsTypeOf(Reset(), UInt(0.W))(_ := false.B))
        verilog shouldNot include("assign")
      }
      they("should match for matched width inputs") {
        val verilog = ChiselStage.emitSystemVerilog(new WireSourceAsTypeOf(Reset(), UInt(1.W))(_ := false.B))
        verilog should include("wire out = in;")
      }
      they("should work for inferred width outputs") {
        val verilog = ChiselStage.emitSystemVerilog(new WireSourceAsTypeOf(Reset(), UInt())(_ := false.B))
        verilog should include("wire out = in;")
      }
    }
  }

}
