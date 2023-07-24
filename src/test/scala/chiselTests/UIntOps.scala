// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
import org.scalatest.matchers.should.Matchers

class UIntOps extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val addout = Output(UInt(32.W))
    val subout = Output(UInt(32.W))
    val addampout = Output(UInt(33.W))
    val subampout = Output(UInt(33.W))
    val timesout = Output(UInt(32.W))
    val divout = Output(UInt(32.W))
    val modout = Output(UInt(32.W))
    val lshiftout = Output(UInt(32.W))
    val rshiftout = Output(UInt(32.W))
    val lrotateout = Output(UInt(32.W))
    val rrotateout = Output(UInt(32.W))
    val lessout = Output(Bool())
    val greatout = Output(Bool())
    val eqout = Output(Bool())
    val noteqout = Output(Bool())
    val lesseqout = Output(Bool())
    val greateqout = Output(Bool())
  })

  dontTouch(io)

  val a = io.a
  val b = io.b

  io.addout := a +% b
  io.subout := a -% b
  io.addampout := a +& b
  io.subampout := a -& b
  io.timesout := (a * b)(31, 0)
  io.divout := a / Mux(b === 0.U, 1.U, b)
  io.modout := a % b
  io.lshiftout := (a << b(3, 0))(31, 0)
  io.rshiftout := a >> b
  io.lrotateout := a.rotateLeft(5)
  io.rrotateout := a.rotateRight(5)
  io.lessout := a < b
  io.greatout := a > b
  io.eqout := a === b
  io.noteqout := (a =/= b)
  io.lesseqout := a <= b
  io.greateqout := a >= b
}

// Note a and b need to be "safe"
class UIntOpsTester(a: Long, b: Long) extends BasicTester {
  require(a >= 0 && b >= 0)

  val dut = Module(new UIntOps)
  dut.io.a := a.asUInt(32.W)
  dut.io.b := b.asUInt(32.W)

  assert(dut.io.addout === (a + b).U(32.W))
  assert(dut.io.subout === (a - b).S(32.W).asUInt)
  assert(dut.io.addampout === (a + b).U(33.W))
  assert(dut.io.subampout === (a - b).S(33.W).asUInt)
  assert(dut.io.timesout === (a * b).U(32.W))
  assert(dut.io.divout === (a / (b.max(1))).U(32.W))
  assert(dut.io.modout === (a % (b.max(1))).U(32.W))
  assert(dut.io.lshiftout === (a << (b % 16)).U(32.W))
  assert(dut.io.rshiftout === (a >> b).U(32.W))
  assert(
    dut.io.lrotateout === s"h${Integer.rotateLeft(a.toInt, 5).toHexString}"
      .U(32.W)
  )
  assert(
    dut.io.rrotateout === s"h${Integer.rotateRight(a.toInt, 5).toHexString}"
      .U(32.W)
  )
  assert(dut.io.lessout === (a < b).B)
  assert(dut.io.greatout === (a > b).B)
  assert(dut.io.eqout === (a == b).B)
  assert(dut.io.noteqout === (a != b).B)
  assert(dut.io.lesseqout === (a <= b).B)
  assert(dut.io.greateqout === (a >= b).B)

  val zeroWidthWire = dut.io.greatout.take(0)
  assert(zeroWidthWire.getWidth == 0, "take(0) should return a zero width")

  val oneWidthWire = dut.io.greatout.take(1)
  assert(oneWidthWire.getWidth == 1, "take(1) should return a one width")

  stop()
}

class GoodBoolConversion extends Module {
  val io = IO(new Bundle {
    val u = Input(UInt(1.W))
    val b = Output(Bool())
  })
  io.b := io.u.asBool
}

class BadBoolConversion extends Module {
  val io = IO(new Bundle {
    val u = Input(UInt(5.W))
    val b = Output(Bool())
  })
  io.b := io.u.asBool
}

class ZeroWidthBoolConversion extends Module {
  val io = IO(new Bundle {
    val u = Input(UInt(0.W))
    val b = Output(Bool())
  })
  io.b := io.u.asBool
}

class NegativeShift(t: => Bits) extends Module {
  val io = IO(new Bundle {})
  Reg(t) >> -1
}

class BasicRotate extends BasicTester {
  val shiftAmount = random.LFSR(4)
  val ctr = RegInit(0.U(4.W))

  val rotL = 1.U(3.W).rotateLeft(shiftAmount)
  val rotR = 1.U(3.W).rotateRight(shiftAmount)

  printf("Shift amount: %d rotateLeft:%b rotateRight:%b\n", shiftAmount, rotL, rotR)

  switch(shiftAmount % 3.U) {
    is(0.U, 3.U) {
      assert(rotL === "b001".U)
      assert(rotR === "b001".U)
    }
    is(1.U) {
      assert(rotL === "b010".U)
      assert(rotR === "b100".U)
    }
    is(2.U) {
      assert(rotL === "b100".U)
      assert(rotR === "b010".U)
    }
  }

  ctr := ctr + 1.U

  when(ctr === 15.U) {
    stop()
  }
}

/** rotating a w-bit word left by n should be equivalent to rotating it by w - n
  * to the left
  */
class MatchedRotateLeftAndRight(w: Int = 13) extends BasicTester {
  val initValue = BigInt(w, scala.util.Random)
  println(s"Initial value: ${initValue.toString(2)}")

  val maxWidthBits = log2Ceil(w + 1)
  val shiftAmount1 = RegInit(0.U(w.W))
  val shiftAmount2 = RegInit(w.U(w.W))
  shiftAmount1 := shiftAmount1 + 1.U
  shiftAmount2 := shiftAmount2 - 1.U

  val value = RegInit(initValue.U(w.W))

  val out1 = value.rotateLeft(shiftAmount1)
  val out2 = value.rotateRight(shiftAmount2)

  printf("rotateLeft by %d: %b\n", shiftAmount1, out1)

  assert(out1 === out2)
  when(shiftAmount1 === w.U) {
    assert(out1 === initValue.U)
    stop()
  }
}

class UIntLitExtractTester extends BasicTester {
  assert("b101010".U.extract(2) === false.B)
  assert("b101010".U.extract(3) === true.B)
  assert("b101010".U.extract(100) === false.B)
  assert("b101010".U(3, 0) === "b1010".U)
  assert("b101010".U(9, 0) === "b0000101010".U)

  assert("b101010".U(6.W)(2) === false.B)
  assert("b101010".U(6.W)(3) === true.B)
  assert("b101010".U(6.W)(100) === false.B)
  assert("b101010".U(6.W)(3, 0) === "b1010".U)
  assert("b101010".U(6.W)(9, 0) === "b0000101010".U)
  stop()
}

class UIntLitZeroWidthTester extends BasicTester {
  assert(-0.U(0.W) === 0.U)
  assert(~0.U(0.W) === 0.U)
  assert(0.U(0.W) + 0.U(0.W) === 0.U)
  assert(5.U * 0.U(0.W) === 0.U)
  assert(0.U(0.W) / 5.U === 0.U)
  assert(0.U(0.W).head(0) === 0.U)
  assert(0.U(0.W).tail(0) === 0.U)
  assert(0.U(0.W).pad(2) === 0.U)
  assert("b0".U(0.W)(0, 0) === "b0".U)
  stop()
}

class UIntOpsSpec extends ChiselPropSpec with Matchers with Utils {

  property("Bools can be created from 1 bit UInts") {
    ChiselStage.emitCHIRRTL(new GoodBoolConversion)
  }

  property("Bools cannot be created from 0 bit UInts") {
    a[Exception] should be thrownBy extractCause[Exception] { ChiselStage.emitCHIRRTL(new ZeroWidthBoolConversion) }
  }

  property("Bools cannot be created from >1 bit UInts") {
    a[Exception] should be thrownBy extractCause[Exception] { ChiselStage.emitCHIRRTL(new BadBoolConversion) }
  }

  property("Out-of-bounds extraction from known-width UInts") {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new RawModule {
        val u = IO(Input(UInt(2.W)))
        u(2, 1)
      })
    }
  }

  property("Out-of-bounds single-bit extraction from known-width UInts") {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new RawModule {
        val u = IO(Input(UInt(2.W)))
        u(2)
      })
    }
  }

  property("Out-of-bounds extraction from known-zero-width UInts") {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new RawModule {
        val u = IO(Input(UInt(0.W)))
        u(0, 0)
      })
    }
  }

  property("Out-of-bounds single-bit extraction from known-zero-width UInts") {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new RawModule {
        val u = IO(Input(UInt(0.W)))
        u(0)
      })
    }
  }

  property("UIntOps should elaborate") {
    ChiselStage.emitCHIRRTL { new UIntOps }
  }

  property("UIntOpsTester should return the correct result") {
    assertTesterPasses { new UIntOpsTester(123, 7) }
  }

  property("Negative shift amounts are invalid") {
    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new NegativeShift(UInt()))
    }
  }

  property("rotateLeft and rotateRight should work for dynamic shift values") {
    assertTesterPasses(new BasicRotate)
  }

  property(
    "rotateLeft and rotateRight should be consistent for dynamic shift values"
  ) {
    assertTesterPasses(new MatchedRotateLeftAndRight)
  }

  property("Bit extraction on literals should work for all non-negative indices") {
    assertTesterPasses(new UIntLitExtractTester)
  }

  property("Basic arithmetic and bit operations with zero-width literals should return correct result (0)") {
    assertTesterPasses(new UIntLitZeroWidthTester)
  }

  property("asBools should support chained apply") {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(Bool())
      })
      io.out := io.in.asBools(2)
    })
  }

  // We use WireDefault with 2 arguments because of
  // https://www.chisel-lang.org/api/3.4.1/chisel3/WireDefault$.html
  //   Single Argument case 2
  property("modulo divide should give min width of arguments") {
    assertKnownWidth(4) {
      val x = WireDefault(UInt(8.W), DontCare)
      val y = WireDefault(UInt(4.W), DontCare)
      val op = x % y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(4) {
      val x = WireDefault(UInt(4.W), DontCare)
      val y = WireDefault(UInt(8.W), DontCare)
      val op = x % y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(UInt(0.W), DontCare)
      val y = WireDefault(UInt(8.W), DontCare)
      val op = x % y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(UInt(8.W), DontCare)
      val y = WireDefault(UInt(0.W), DontCare)
      val op = x % y
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("division should give the width of the numerator") {
    assertKnownWidth(8) {
      val x = WireDefault(UInt(8.W), DontCare)
      val y = WireDefault(UInt(4.W), DontCare)
      val op = x / y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(4) {
      val x = WireDefault(UInt(4.W), DontCare)
      val y = WireDefault(UInt(8.W), DontCare)
      val op = x / y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(UInt(0.W), DontCare)
      val y = WireDefault(UInt(8.W), DontCare)
      val op = x / y
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("head and tail should be zero-width on zero-width wires") {
    assertKnownWidth(0) {
      val x = WireDefault(UInt(0.W), DontCare)
      val op = x.tail(0)
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(UInt(0.W), DontCare)
      val op = x.head(0)
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("basic arithmetic operations be supported on zero-width wires") {
    assertKnownWidth(0) {
      val x = WireDefault(UInt(0.W), DontCare)
      val y = WireDefault(UInt(0.W), DontCare)
      val op = x + y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(UInt(0.W), DontCare)
      val y = WireDefault(UInt(0.W), DontCare)
      val op = x - y
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(UInt(0.W), DontCare)
      val y = WireDefault(UInt(0.W), DontCare)
      val op = x * y
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("Zero-width bit extractions should be supported") {
    assertKnownWidth(0) {
      val x = WireDefault(UInt(8.W), DontCare)
      val op = x(-1, 0)
      WireDefault(chiselTypeOf(op), op)
    }
    assertKnownWidth(0) {
      val x = WireDefault(UInt(8.W), DontCare)
      val hi = 5
      val lo = 6
      val op = (x >> lo)(hi - lo, 0)
      WireDefault(chiselTypeOf(op), op)
    }
  }

  property("Zero-width bit extractions from the middle of a UInt should give an actionable error") {
    val (log, x) = grabLog(intercept[Exception](ChiselStage.emitCHIRRTL(new RawModule {
      val x = WireDefault(UInt(8.W), DontCare)
      val op = x(5, 6)
      WireDefault(chiselTypeOf(op), op)
    })))
    log should include(
      "Invalid bit range [hi=5, lo=6]. If you are trying to extract zero-width range, right-shift by 'lo' before extracting."
    )
  }

  property("emit warning if dynamic index is too wide or too narrow") {
    class TooWide extends Module {
      val in = IO(Input(UInt(2.W)))
      val index = IO(Input(UInt(2.W)))
      val out = IO(Output(Bool()))
      out := in(index)
    }
    class TooNarrow extends Module {
      val in = IO(Input(UInt(3.W)))
      val index = IO(Input(UInt(1.W)))
      val out = IO(Output(Bool()))
      out := in(index)
    }

    Seq(
      grabLog(ChiselStage.emitCHIRRTL(new TooWide)),
      grabLog(ChiselStage.emitCHIRRTL(new TooNarrow))
    ).foreach {
      case (log, _) =>
        log should include("warn")
    }

    a[ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(new RawModule {
        val in = IO(Input(UInt(0.W)))
        val index = IO(Input(UInt(1.W)))
        val out = IO(Output(Bool()))
        out := in(index)
      })
    }

    class Ok extends Module {
      val in1 = IO(Input(UInt(9.W)))
      val index1 = IO(Input(UInt(4.W)))
      val out1 = IO(Output(Bool()))
      out1 := in1(index1)

      val in2 = IO(Input(UInt(8.W)))
      val index2 = IO(Input(UInt(3.W)))
      val out2 = IO(Output(Bool()))
      out2 := in2(index2)

      val in3 = IO(Input(Bool()))
      val index3 = IO(Input(UInt(0.W)))
      val out3 = IO(Output(Bool()))
      out3 := in3(index3)
    }

    Seq(
      grabLog(ChiselStage.emitCHIRRTL(new Ok))
    ).foreach {
      case (log, _) =>
        log should be("")
    }
  }
}
