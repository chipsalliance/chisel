// See LICENSE for license details.

package chiselTests
import Chisel._
import org.scalatest._
import Chisel.testers.BasicTester

class UIntOps extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(16))
    val b = Input(UInt(16))
    val addout = Output(UInt(16))
    val subout = Output(UInt(16))
    val timesout = Output(UInt(16))
    val divout = Output(UInt(16))
    val modout = Output(UInt(16))
    val lshiftout = Output(UInt(16))
    val rshiftout = Output(UInt(16))
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
  io.timesout := (a * b)(15, 0)
  io.divout := a / Mux(b === UInt(0), UInt(1), b)
  // io.modout := a % b
  // TODO:
  io.modout := UInt(0)
  io.lshiftout := (a << b(3, 0))(15, 0)
  io.rshiftout := a >> b
  io.lessout := a < b
  io.greatout := a > b
  io.eqout := a === b
  io.noteqout := (a != b)
  io.lesseqout := a <= b
  io.greateqout := a >= b
}

/*
class UIntOpsTester(c: UIntOps) extends Tester(c) {
  def uintExpect(d: Bits, x: BigInt) {
    val mask = (1 << 16) - 1
    println(" E = " + x + " X&M = " + (x & mask))
    expect(d, x & mask)
  }
  for (t <- 0 until 16) {
    val test_a = rnd.nextInt(1 << 16)
    val test_b = rnd.nextInt(1 << 16)
    println("A = " + test_a + " B = " + test_b)
    poke(c.io.a, test_a)
    poke(c.io.b, test_b)
    step(1)
    uintExpect(c.io.addout, test_a + test_b)
    uintExpect(c.io.subout, test_a - test_b)
    uintExpect(c.io.divout, if (test_b == 0) 0 else test_a / test_b)
    uintExpect(c.io.timesout, test_a * test_b)
    // uintExpect(c.io.modout, test_a % test_b)
    uintExpect(c.io.lshiftout, test_a << (test_b&15))
    uintExpect(c.io.rshiftout, test_a >> test_b)
    expect(c.io.lessout, int(test_a < test_b))
    expect(c.io.greatout, int(test_a > test_b))
    expect(c.io.eqout, int(test_a == test_b))
    expect(c.io.noteqout, int(test_a != test_b))
    expect(c.io.lessout, int(test_a <= test_b))
    expect(c.io.greateqout, int(test_a >= test_b))
  }
}
*/

class GoodBoolConversion extends Module {
  val io = IO(new Bundle {
    val u = Input(UInt(1))
    val b = Output(Bool())
  })
  io.b := io.u.toBool
}

class BadBoolConversion extends Module {
  val io = IO(new Bundle {
    val u = Input(UInt(width = 5))
    val b = Output(Bool())
  })
  io.b := io.u.toBool
}

class UIntOpsSpec extends ChiselPropSpec with Matchers {
  property("Bools can be created from 1 bit UInts") {
    elaborate(new GoodBoolConversion)
  }

  property("Bools cannot be created from >1 bit UInts") {
    a [Exception] should be thrownBy { elaborate(new BadBoolConversion) }
  }

  property("UIntOps should elaborate") {
    elaborate { new UIntOps }
  }

  ignore("UIntOpsTester should return the correct result") { }
}

