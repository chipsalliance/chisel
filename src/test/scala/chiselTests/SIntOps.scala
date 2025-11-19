// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import circt.stage.ChiselStage
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers

class SIntOps extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(16.W))
    val b = Input(SInt(16.W))
    val addout = Output(SInt(16.W))
    val subout = Output(SInt(16.W))
    val timesout = Output(SInt(16.W))
    val divout = Output(SInt(16.W))
    val modout = Output(SInt(16.W))
    val lshiftout = Output(SInt(16.W))
    val rshiftout = Output(SInt(16.W))
    val lessout = Output(Bool())
    val greatout = Output(Bool())
    val eqout = Output(Bool())
    val noteqout = Output(Bool())
    val lesseqout = Output(Bool())
    val greateqout = Output(Bool())
    val negout = Output(SInt(16.W))
  })

  val a = io.a
  val b = io.b

  io.addout := a +% b
  io.subout := a -% b
  // TODO:
  // io.timesout := (a * b)(15, 0)
  // io.divout := a / Mux(b === 0.S, 1.S, b)
  // io.divout := (a / b)(15, 0)
  // io.modout := 0.S
  // io.lshiftout := (a << 12)(15, 0) //  (a << ub(3, 0))(15, 0).toSInt
  io.rshiftout := (a >> 8) // (a >> ub).toSInt
  io.lessout := a < b
  io.greatout := a > b
  io.eqout := a === b
  io.noteqout := (a =/= b)
  io.lesseqout := a <= b
  io.greateqout := a >= b
  io.negout := -a(15, 0).asSInt
  io.negout := (0.S -% a)
}

/*
class SIntOpsTester(c: SIntOps) extends Tester(c) {
  def sintExpect(d: Bits, x: BigInt) {
    val mask = (1 << 16) - 1
    val sbit = (1 << 15)
    val y = x & mask
    val r = if ((y & sbit) == 0) y else (-(~y)-1)
    expect(d, r)
  }
  for (t <- 0 until 16) {
    val test_a = (1 << 15) - rnd.nextInt(1 << 16)
    val test_b = (1 << 15) - rnd.nextInt(1 << 16)
    poke(c.io.a, test_a)
    poke(c.io.b, test_b)
    step(1)
    sintExpect(c.io.addout, test_a + test_b)
    sintExpect(c.io.subout, test_a - test_b)
    sintExpect(c.io.timesout, test_a * test_b)
    // sintExpect(c.io.divout, if (test_b == 0) 0 else test_a / test_b)
    sintExpect(c.io.divout, test_a * test_b)
    // sintExpect(c.io.modout, test_a % test_b)
    // sintExpect(c.io.lshiftout, test_a << (test_b&15))
    // sintExpect(c.io.rshiftout, test_a >> test_b)
    sintExpect(c.io.lshiftout, test_a << 12)
    sintExpect(c.io.rshiftout, test_a >> 8)
    sintExpect(c.io.negout, -test_a)
    expect(c.io.lessout, int(test_a < test_b))
    expect(c.io.greatout, int(test_a > test_b))
    expect(c.io.eqout, int(test_a == test_b))
    expect(c.io.noteqout, int(test_a != test_b))
    expect(c.io.lessout, int(test_a <= test_b))
    expect(c.io.greateqout, int(test_a >= test_b))
  }
}
 */

class SIntLitExtractTester extends Module {
  assert(-5.S.extract(1) === true.B)
  assert(-5.S.extract(2) === false.B)
  assert(-5.S.extract(100) === true.B)
  assert(-5.S(3, 0) === "b1011".U)
  assert(-5.S(9, 0) === "b1111111011".U)
  assert(-5.S(4.W)(1) === true.B)
  assert(-5.S(4.W)(2) === false.B)
  assert(-5.S(4.W)(100) === true.B)
  assert(-5.S(4.W)(3, 0) === "b1011".U)
  assert(-5.S(4.W)(9, 0) === "b1111111011".U)
  stop()
}

class SIntLitZeroWidthTester extends Module {
  assert(-0.S(0.W) === 0.S)
  assert(~0.S(0.W) === 0.S)
  assert(0.S(0.W) + 0.S(0.W) === 0.S)
  assert(5.S * 0.S(0.W) === 0.S)
  assert(0.S(0.W) / 5.S === 0.S)
  assert(0.S(0.W).head(0) === 0.U)
  assert(0.S(0.W).tail(0) === 0.U)
  assert(0.S(0.W).pad(1) === 0.S)
  assert(-0.S(0.W)(0, 0) === "b0".U)
  stop()
}

class SIntOpsSpec extends AnyPropSpec with Matchers with ShiftRightWidthBehavior with ChiselSim with LogUtils {

  property("SIntOps should elaborate") {
    ChiselStage.emitCHIRRTL { new SIntOps }
  }

  property("Negative shift amounts are invalid") {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new NegativeShift(SInt()))
    }
  }

  ignore("SIntOpsTester should return the correct result") {}

  property("Bit extraction on literals should work for all non-negative indices") {
    simulate(new SIntLitExtractTester)(RunUntilFinished(3))
  }

  property("Basic arithmetic and bit operations with zero-width literals should return correct result (0)") {
    simulate(new SIntLitZeroWidthTester)(RunUntilFinished(3))
  }

  // We use WireDefault with 2 arguments because of
  // https://www.chisel-lang.org/api/3.4.1/chisel3/WireDefault$.html
  //   Single Argument case 2
  property("modulo divide should give min width of arguments") {
    assertKnownWidth(4) {
      val x = WireDefault(SInt(8.W), DontCare)
      val y = WireDefault(SInt(4.W), DontCare)
      val op = x % y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(4) {
      val x = WireDefault(SInt(4.W), DontCare)
      val y = WireDefault(SInt(8.W), DontCare)
      val op = x % y
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("division should give the width of the numerator + 1") {
    assertKnownWidth(9) {
      val x = WireDefault(SInt(8.W), DontCare)
      val y = WireDefault(SInt(4.W), DontCare)
      val op = x / y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(5) {
      val x = WireDefault(SInt(4.W), DontCare)
      val y = WireDefault(SInt(8.W), DontCare)
      val op = x / y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(1) {
      val x = WireDefault(SInt(0.W), DontCare)
      val y = WireDefault(SInt(8.W), DontCare)
      val op = x / y
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("Zero-width bit extractions should be supported") {
    assertKnownWidth(0) {
      val x = WireDefault(SInt(8.W), DontCare)
      val op = x(-1, 0)
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(SInt(8.W), DontCare)
      val hi = 5
      val lo = 6
      val op = (x >> lo)(hi - lo, 0)
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("Zero-width bit extractions from the middle of an SInt should give an actionable error") {
    val (log, x) = grabLog(intercept[Exception](ChiselStage.emitCHIRRTL(new RawModule {
      val x = WireDefault(SInt(8.W), DontCare)
      val op = x(5, 6)
      WireDefault(chiselTypeOf(op), op)
    })))
    log should include(
      "Invalid bit range [hi=5, lo=6]. If you are trying to extract zero-width range, right-shift by 'lo' before extracting."
    )
  }

  property("Static right-shift should have a minimum width of 1") {
    testSIntShiftRightWidthBehavior(chiselMinWidth = 1, firrtlMinWidth = 1)
  }

  property("Static right-shift should have width of 0 in Chisel and 1 in FIRRTL with --use-legacy-width") {
    val args = Array("--use-legacy-width")

    testSIntShiftRightWidthBehavior(chiselMinWidth = 0, firrtlMinWidth = 1, args = args)

    // Focused test to show the mismatch
    class TestModule extends Module {
      val in = IO(Input(SInt(8.W)))
      val widthcheck = Wire(SInt())
      val shifted = in >> 8
      shifted.getWidth should be(0)
      widthcheck := shifted
      dontTouch(widthcheck)
    }
    val verilog = ChiselStage.emitSystemVerilog(new TestModule, args)
    verilog should include(" widthcheck = in[7];")
  }

  property("Calling .asUInt on an SInt literal should maintain the literal value") {
    val s0 = 3.S
    val u0 = s0.asUInt
    u0.litValue should be(3)

    val s1 = -3.S
    val u1 = s1.asUInt
    u1.litValue should be(5)

    val s2 = -3.S(8.W)
    val u2 = s2.asUInt
    u2.litValue should be(0xfd)

    simulate {
      new Module {
        // Check that it gives the same value as the generated hardware
        val wire0 = WireInit(s0).asUInt
        chisel3.assert(u0.litValue.U === wire0)
        val wire1 = WireInit(s1).asUInt
        chisel3.assert(u1.litValue.U === wire1)
        val wire2 = WireInit(s2).asUInt
        chisel3.assert(u2.litValue.U === wire2)

        stop()
      }
    }(RunUntilFinished(3))
  }

  property("Calling .asSInt on a SInt literal should maintain the literal value") {
    3.S.asSInt.litValue should be(3)
    -5.S.asSInt.litValue should be(-5)
  }

  property("Calling .pad on a SInt literal should maintain the literal value") {
    -5.S.getWidth should be(4)
    -5.S.pad(2).litValue should be(-5)
    -5.S.pad(2).getWidth should be(4)
    -5.S.pad(4).litValue should be(-5)
    -5.S.pad(4).getWidth should be(4)
    -5.S.pad(6).litValue should be(-5)
    -5.S.pad(6).getWidth should be(6)

    -5.S(8.W).getWidth should be(8)
    -5.S(8.W).pad(2).litValue should be(-5)
    -5.S(8.W).pad(2).getWidth should be(8)
    -5.S(8.W).pad(8).litValue should be(-5)
    -5.S(8.W).pad(8).getWidth should be(8)
    -5.S(8.W).pad(16).litValue should be(-5)
    -5.S(8.W).pad(16).getWidth should be(16)
  }

  property("Casting a SInt literal to a Bundle should maintain the literal value") {
    class SimpleBundle extends Bundle {
      val x = UInt(4.W)
      val y = UInt(4.W)
    }
    val blit = -23.S.asTypeOf(new SimpleBundle)
    blit.litOption should be(Some(0x29))
    blit.x.litOption should be(Some(2))
    blit.y.litOption should be(Some(9))
  }

  property("SInt literals with too small of a width should be rejected") {
    // Sanity checks.
    0.S.getWidth should be(1)
    0.S(0.W).getWidth should be(0)
    -1.S.getWidth should be(1)
    1.S.getWidth should be(2)
    // The real check.
    -2.S.getWidth should be(2)
    an[IllegalArgumentException] shouldBe thrownBy(-2.S(1.W))
    0xde.S.getWidth should be(9)
    an[IllegalArgumentException] shouldBe thrownBy(0xde.S(8.W))
    an[IllegalArgumentException] shouldBe thrownBy(0xde.S(4.W))
  }
}
