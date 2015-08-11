package ChiselTests
import Chisel._
import Chisel.testers._

class Inc extends Module {
  val io = new Bundle {
    val in  = UInt(INPUT,  32)
    val out = UInt(OUTPUT, 32)
  }
  io.out := io.in + UInt(1)
}

class ModuleWire extends Module {
  val io = new Bundle {
    val in  = UInt(INPUT,  32)
    val out = UInt(OUTPUT, 32)
  }
  val inc = Module(new Inc).io 
  inc.in := io.in
  io.out := inc.out 
}


class ModuleWireTester(c: ModuleWire) extends Tester(c) {
  for (t <- 0 until 16) {
    val test_in = rnd.nextInt(256)
    poke(c.io.in, test_in)
    step(1)
    expect(c.io.out, test_in + 1)
  }
}
