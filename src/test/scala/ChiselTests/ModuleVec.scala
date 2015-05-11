package ChiselTests
import Chisel._

class PlusOne extends Module {
  val io = new Bundle {
    val in  = UInt(INPUT,  32)
    val out = UInt(OUTPUT, 32)
  }
  io.out := io.in + UInt(1)
}

class ModuleVec(val n: Int) extends Module {
  val io = new Bundle {
    val ins  = Vec(UInt(INPUT,  32), n)
    val outs = Vec(UInt(OUTPUT, 32), n)
  }
  val pluses = Vec.fill(n){ Module(new PlusOne).io }
  for (i <- 0 until n) {
    pluses(i).in := io.ins(i)
    io.outs(i)   := pluses(i).out 
  }
}


class ModuleVecTester(c: ModuleVec) extends Tester(c) {
  for (t <- 0 until 16) {
    val test_ins = Array.fill(c.n){ rnd.nextInt(256) }
    for (i <- 0 until c.n) 
      poke(c.io.ins(i), test_ins(i))
    step(1)
    for (i <- 0 until c.n) 
      expect(c.io.outs(i), test_ins(i) + 1)
  }
}
