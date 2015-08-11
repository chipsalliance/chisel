package ChiselTests
import Chisel._
import Chisel.testers._

class Rom extends Module {
  val io = new Bundle {
    val addr = UInt(INPUT,  4)
    val out  = UInt(OUTPUT, 5)
  }
  val r = Vec(Range(0, 1 << 4).map(i => UInt(i * 2, width = 5)))
  io.out := r(io.addr)
}


class RomTester(c: Rom) extends Tester(c) {
  val r = Array.tabulate(1 << 4){ i => i * 2}
  for (i <- 0 until 10) {
    val a = rnd.nextInt(1 << 4)
    poke(c.io.addr, a)
    step(1)
    expect(c.io.out, r(a))
  }

}
