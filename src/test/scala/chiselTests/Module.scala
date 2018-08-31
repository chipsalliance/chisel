// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{withClock, withReset}

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
  val pluses = VecInit(Seq.fill(n){ Module(new PlusOne).io })
  for (i <- 0 until n) {
    pluses(i).in := io.ins(i)
    io.outs(i)   := pluses(i).out
  }
}

class ModuleWire extends Module {
  val io = IO(new SimpleIO)
  val inc = Wire(chiselTypeOf(Module(new PlusOne).io))
  inc.in := io.in
  io.out := inc.out
}

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

  property("object Module.clock should return a reference to the currently in scope clock") {
    elaborate(new Module {
      val io = IO(new Bundle {
        val clock2 = Input(Clock())
      })
      assert(Module.clock eq this.clock)
      withClock(io.clock2) { assert(Module.clock eq io.clock2) }
    })
  }
  property("object Module.reset should return a reference to the currently in scope reset") {
    elaborate(new Module {
      val io = IO(new Bundle {
        val reset2 = Input(Bool())
      })
      assert(Module.reset eq this.reset)
      withReset(io.reset2) { assert(Module.reset eq io.reset2) }
    })
  }
  property("object Module.currentModule should return an Option reference to the current Module") {
    def checkModule(mod: Module): Boolean = Module.currentModule.map(_ eq mod).getOrElse(false)
    elaborate(new Module {
      val io = IO(new Bundle { })
      assert(Module.currentModule.get eq this)
      assert(checkModule(this))
    })
  }
}
