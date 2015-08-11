package ChiselTests
import Chisel._
import Chisel.testers._

class Tbl extends Module {
  val io = new Bundle {
    val  i  = Bits(INPUT,  16)
    val we  = Bool(INPUT)
    val  d  = Bits(INPUT,  16)
    val  o  = Bits(OUTPUT, 16)
  }
  val m = Mem(Bits(width = 10), 256)
  io.o := Bits(0)
  when (io.we) { m(io.i) := io.d(9, 0) }
  .otherwise   { io.o := m(io.i) }
}

class TblTester(c: Tbl) extends Tester(c) {
  val m = Array.fill(1 << 16){ 0 }
  for (t <- 0 until 16) {
    val i  = rnd.nextInt(1 << 16)
    val d  = rnd.nextInt(1 << 16)
    val we = rnd.nextInt(2)
    poke(c.io.i,   i)
    poke(c.io.we, we)
    poke(c.io.d,   d)
    step(1)
    expect(c.io.o, if (we == 1) 0 else m(i))
    if (we == 1)
      m(i) = d
  }
}
