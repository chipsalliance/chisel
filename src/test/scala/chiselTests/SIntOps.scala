// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester

class SIntOps extends Module {
  val io = new Bundle {
    val a = SInt(INPUT, 16)
    val b = SInt(INPUT, 16)
    val addout = SInt(OUTPUT, 16)
    val subout = SInt(OUTPUT, 16)
    val timesout = SInt(OUTPUT, 16)
    val divout = SInt(OUTPUT, 16)
    val modout = SInt(OUTPUT, 16)
    val lshiftout = SInt(OUTPUT, 16)
    val rshiftout = SInt(OUTPUT, 16)
    val lessout = Bool(OUTPUT)
    val greatout = Bool(OUTPUT)
    val eqout = Bool(OUTPUT)
    val noteqout = Bool(OUTPUT)
    val lesseqout = Bool(OUTPUT)
    val greateqout = Bool(OUTPUT)
    val negout = SInt(OUTPUT, 16)
  }

  val a = io.a
  val b = io.b

  io.addout := a +% b
  io.subout := a -% b
  // TODO:
  //io.timesout := (a * b)(15, 0)
  //io.divout := a / Mux(b === SInt(0), SInt(1), b)
  //io.divout := (a / b)(15, 0)
  //io.modout := SInt(0)
  //io.lshiftout := (a << 12)(15, 0) //  (a << ub(3, 0))(15, 0).toSInt
  io.rshiftout := (a >> 8) // (a >> ub).toSInt
  io.lessout := a < b
  io.greatout := a > b
  io.eqout := a === b
  io.noteqout := (a != b)
  io.lesseqout := a <= b
  io.greateqout := a >= b
  // io.negout := -a(15, 0).toSInt
  io.negout := (SInt(0) -% a)
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

class SIntOpsSpec extends ChiselPropSpec {

  property("SIntOps should elaborate") {
    elaborate { new SIntOps }
  }

  ignore("SIntOpsTester should return the correct result") { }
}
