package ChiselTests
import Chisel._

class GCD extends Module {
  val io = new Bundle {
    val a  = Bits(INPUT,  16)
    val b  = Bits(INPUT,  16)
    val e  = Bool(INPUT)
    val z  = Bits(OUTPUT, 16)
    val v  = Bool(OUTPUT)
  }
  val x = Reg(Bits(width = 16))
  val y = Reg(Bits(width = 16))
  when (x > y)   { x := x -% y }
  .otherwise     { y := y -% x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === Bits(0)
}

class GCDTester(c: GCD) extends Tester(c) {
  val (a, b, z) = (64, 48, 16)
  do {
    val first = if (t == 0) 1 else 0;
    poke(c.io.a, a)
    poke(c.io.b, b)
    poke(c.io.e, first)
    step(1)
  } while (t <= 1 || peek(c.io.v) == 0)
  expect(c.io.z, z)
}
