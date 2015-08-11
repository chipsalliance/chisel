package ChiselTests
import Chisel._
import Chisel.testers._

class BitsOps extends Module {
  val io = new Bundle {
    val a = Bits(INPUT, 16)
    val b = Bits(INPUT, 16)
    val notout = Bits(OUTPUT, 16)
    val andout = Bits(OUTPUT, 16)
    val orout = Bits(OUTPUT, 16)
    val xorout = Bits(OUTPUT, 16)
  }

  io.notout := ~io.a
  io.andout := io.a & io.b
  io.orout := io.a | io.b
  io.xorout := io.a ^ io.b
}

class BitsOpsTester(c: BitsOps) extends Tester(c) {
  val mask = (1 << 16)-1;
  for (t <- 0 until 16) {
    val test_a = rnd.nextInt(1 << 16)
    val test_b = rnd.nextInt(1 << 16)
    poke(c.io.a, test_a)
    poke(c.io.b, test_b)
    step(1)
    expect(c.io.notout, mask & (~test_a))
    expect(c.io.andout, mask & (test_a & test_b))
    expect(c.io.orout,  mask & (test_a | test_b))
    expect(c.io.xorout, mask & (test_a ^ test_b))
  }
}
