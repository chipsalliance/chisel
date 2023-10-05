// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import circt.stage.ChiselStage

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
  //io.timesout := (a * b)(15, 0)
  //io.divout := a / Mux(b === 0.S, 1.S, b)
  //io.divout := (a / b)(15, 0)
  //io.modout := 0.S
  //io.lshiftout := (a << 12)(15, 0) //  (a << ub(3, 0))(15, 0).toSInt
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

class SIntLitExtractTester extends BasicTester {
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

class SIntLitZeroWidthTester extends BasicTester {
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

class SIntOpsSpec extends ChiselPropSpec with Utils {

  property("SIntOps should elaborate") {
    ChiselStage.emitCHIRRTL { new SIntOps }
  }

  property("Negative shift amounts are invalid") {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new NegativeShift(SInt()))
    }
  }

  ignore("SIntOpsTester should return the correct result") {}

  property("Bit extraction on literals should work for all non-negative indices") {
    assertTesterPasses(new SIntLitExtractTester)
  }

  property("Basic arithmetic and bit operations with zero-width literals should return correct result (0)") {
    assertTesterPasses(new SIntLitZeroWidthTester)
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

}
