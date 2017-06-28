// See LICENSE for license details.

package chiselTests

import chisel3._

class SimpleIO extends Bundle {
  val in  = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

class PlusOne extends Module {
  val io = IO(new SimpleIO)
  io.out := io.in + 1.asUInt
}

class ModuleVec(val n: Int) extends Module {
  val io = IO(new Bundle {
    val ins  = Input(Vec(n, UInt(32.W)))
    val outs = Output(Vec(n, UInt(32.W)))
  })
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
  val io = IO(new SimpleIO)
  val inc = Wire(Module(new PlusOne).io.chiselCloneType)
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
  val io = IO(new Bundle {
    val s = new SimpleIO
    val en = Output(Bool())
  })
  when(io.en) {
    val inc = Module(new PlusOne).io
    inc.in := io.s.in
    io.s.out := inc.out
  } otherwise { io.s.out := io.s.in }
}

class ModuleForgetWrapper extends Module {
  val io = IO(new SimpleIO)
  val inst = new PlusOne
}

class ModuleDoubleWrap extends Module {
  val io = IO(new SimpleIO)
  val inst = Module(Module(new PlusOne))
}

class ModuleRewrap extends Module {
  val io = IO(new SimpleIO)
  val inst = Module(new PlusOne)
  val inst2 = Module(inst)
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

  property("Forgetting a Module() wrapper should result in an error") {
    (the [ChiselException] thrownBy {
      elaborate { new ModuleForgetWrapper }
    }).getMessage should include("attempted to instantiate a Module without wrapping it")
  }

  property("Double wrapping a Module should result in an error") {
    (the [ChiselException] thrownBy {
      elaborate { new ModuleDoubleWrap }
    }).getMessage should include("Called Module() twice without instantiating a Module")
  }

  property("Rewrapping an already instantiated Module should result in an error") {
    (the [ChiselException] thrownBy {
      elaborate { new ModuleRewrap }
    }).getMessage should include("This is probably due to rewrapping a Module instance")
  }
}
