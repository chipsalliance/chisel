// See LICENSE for license details.

package chiselTests
import Chisel._

class SimpleIO extends Bundle {
  val in  = UInt(INPUT,  32)
  val out = UInt(OUTPUT, 32)
}

class PlusOne extends Module {
  val io = new SimpleIO
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

/*
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
*/

class ModuleWire extends Module {
  val io = new SimpleIO
  val inc = Wire(Module(new PlusOne).io)
  inc.in := io.in
  io.out := inc.out
}

/*
class ModuleWireTester(c: ModuleWire) extends Tester(c) {
  for (t <- 0 until 16) {
    val test_in = rnd.nextInt(256)
    poke(c.io.in, test_in)
    step(1)
    expect(c.io.out, test_in + 1)
  }
}
*/

class ModuleWhen extends Module {
  val io = new Bundle {
    val s = new SimpleIO
    val en = Bool()
  }
  when(io.en) {
    val inc = Module(new PlusOne).io
    inc.in := io.s.in
    io.s.out := inc.out
  } otherwise { io.s.out := io.s.in }
}

class ModuleSpec extends ChiselPropSpec {

  property("ModuleVec should elaborate") {
    elaborate { new ModuleVec(2) }
  }

  ignore("ModuleVecTester should return the correct result") { }

  property("ModuleWire should elaborate") {
    elaborate { new ModuleWire }
  }

  ignore("ModuleWireTester should return the correct result") { }

  property("ModuleWhen should elaborate") {
    elaborate { new ModuleWhen }
  }

  ignore("ModuleWhenTester should return the correct result") { }
}
