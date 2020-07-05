// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import org.scalacheck.Shrink
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
    val lessout = Output(Bool())
    val greatout = Output(Bool())
    val eqout = Output(Bool())
    val noteqout = Output(Bool())
    val lesseqout = Output(Bool())
    val greateqout = Output(Bool())
  })

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
  assert(dut.io.divout === (a / (b max 1)).U(32.W))
  assert(dut.io.modout === (a % (b max 1)).U(32.W))
  assert(dut.io.lshiftout === (a << (b % 16)).U(32.W))
  assert(dut.io.rshiftout === (a >> b).U(32.W))
  assert(dut.io.lessout === (a < b).B)
  assert(dut.io.greatout === (a > b).B)
  assert(dut.io.eqout === (a == b).B)
  assert(dut.io.noteqout === (a != b).B)
  assert(dut.io.lesseqout === (a <= b).B)
  assert(dut.io.greateqout === (a >= b).B)

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

class NegativeShift(t: => Bits) extends Module {
  val io = IO(new Bundle {})
  Reg(t) >> -1
}

class UIntLitExtractTester extends BasicTester {
  assert("b101010".U(2) === false.B)
  assert("b101010".U(3) === true.B)
  assert("b101010".U(100) === false.B)
  assert("b101010".U(3, 0) === "b1010".U)
  assert("b101010".U(9, 0) === "b0000101010".U)

  assert("b101010".U(6.W)(2) === false.B)
  assert("b101010".U(6.W)(3) === true.B)
  assert("b101010".U(6.W)(100) === false.B)
  assert("b101010".U(6.W)(3, 0) === "b1010".U)
  assert("b101010".U(6.W)(9, 0) === "b0000101010".U)
  stop()
}

class UIntOpsSpec extends ChiselPropSpec with Matchers with Utils {
  // Disable shrinking on error.
  implicit val noShrinkListVal = Shrink[List[Int]](_ => Stream.empty)
  implicit val noShrinkInt = Shrink[Int](_ => Stream.empty)

  property("Bools can be created from 1 bit UInts") {
    ChiselStage.elaborate(new GoodBoolConversion)
  }

  property("Bools cannot be created from >1 bit UInts") {
    a [Exception] should be thrownBy extractCause[Exception] { ChiselStage.elaborate(new BadBoolConversion) }
  }

  property("UIntOps should elaborate") {
    ChiselStage.elaborate { new UIntOps }
  }

  property("UIntOpsTester should return the correct result") {
    assertTesterPasses { new UIntOpsTester(123, 7) }
  }

  property("Negative shift amounts are invalid") {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new NegativeShift(UInt()))
    }
  }

  property("Bit extraction on literals should work for all non-negative indices") {
    assertTesterPasses(new UIntLitExtractTester)
  }

  property("asBools should support chained apply") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(Bool())
      })
      io.out := io.in.asBools()(2)
    })
  }
}
