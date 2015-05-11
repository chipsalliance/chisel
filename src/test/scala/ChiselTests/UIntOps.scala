package ChiselTests
import Chisel._

class UIntOps extends Module {
  val io = new Bundle {
    val a = UInt(INPUT, 16)
    val b = UInt(INPUT, 16)
    val addout = UInt(OUTPUT, 16)
    val subout = UInt(OUTPUT, 16)
    val timesout = UInt(OUTPUT, 16)
    val divout = UInt(OUTPUT, 16)
    val modout = UInt(OUTPUT, 16)
    val lshiftout = UInt(OUTPUT, 16)
    val rshiftout = UInt(OUTPUT, 16)
    val lessout = Bool(OUTPUT)
    val greatout = Bool(OUTPUT)
    val eqout = Bool(OUTPUT)
    val noteqout = Bool(OUTPUT)
    val lesseqout = Bool(OUTPUT)
    val greateqout = Bool(OUTPUT)
  }

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

