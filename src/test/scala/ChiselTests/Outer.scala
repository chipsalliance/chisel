package ChiselTests
import Chisel._

class Inner extends Module {
  val io = new Bundle {
    val in  = Bits(INPUT, 8)
    val out = Bits(OUTPUT, 8)
  }
  io.out := io.in +% Bits(1)
}

class Outer extends Module {
  val io = new Bundle { 
    val in  = Bits(INPUT, 8)
    val out = Bits(OUTPUT, 8)
  }
  // val c = Module(new Inner)
  val c = Array(Module(new Inner))
  // val w = Wire(Bits(NO_DIR, 8))
  // w := io.in
  c(0).io.in := io.in
  io.out  := (c(0).io.out * Bits(2))(7,0)
}

class OuterTester(c: Outer) extends Tester(c) {
  for (t <- 0 until 16) {
    val test_in = rnd.nextInt(256)
    poke(c.io.in, test_in)
    step(1)
    expect(c.io.out, ((test_in + 1) * 2)&255)
  }
}
