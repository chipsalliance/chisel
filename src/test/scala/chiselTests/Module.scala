// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.experimental.DataMirror

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

class ModuleWrapper(gen: => Module) extends Module {
  val io = IO(new Bundle{})
  val child = Module(gen)
  override val desiredName = s"${child.desiredName}Wrapper"
}

class NullModuleWrapper extends Module {
  val io = IO(new Bundle{})
  override lazy val desiredName = s"${child.desiredName}Wrapper"
  println(s"My name is ${name}") // scalastyle:ignore regex
  val child = Module(new ModuleWire)
}

class ModuleSpec extends ChiselPropSpec with Utils {

  property("ModuleVec should elaborate") {
    ChiselStage.elaborate { new ModuleVec(2) }
  }

  ignore("ModuleVecTester should return the correct result") { }

  property("ModuleWire should elaborate") {
    ChiselStage.elaborate { new ModuleWire }
  }

  ignore("ModuleWireTester should return the correct result") { }

  property("ModuleWhen should elaborate") {
    ChiselStage.elaborate { new ModuleWhen }
  }

  ignore("ModuleWhenTester should return the correct result") { }

  property("Forgetting a Module() wrapper should result in an error") {
    (the [ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new ModuleForgetWrapper }
    }).getMessage should include("attempted to instantiate a Module without wrapping it")
  }

  property("Double wrapping a Module should result in an error") {
    (the [ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new ModuleDoubleWrap }
    }).getMessage should include("Called Module() twice without instantiating a Module")
  }

  property("Rewrapping an already instantiated Module should result in an error") {
    (the [ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new ModuleRewrap }
    }).getMessage should include("This is probably due to rewrapping a Module instance")
  }

  property("object Module.clock should return a reference to the currently in scope clock") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val clock2 = Input(Clock())
      })
      assert(Module.clock eq this.clock)
      withClock(io.clock2) { assert(Module.clock eq io.clock2) }
    })
  }
  property("object Module.reset should return a reference to the currently in scope reset") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val reset2 = Input(Bool())
      })
      assert(Module.reset eq this.reset)
      withReset(io.reset2) { assert(Module.reset eq io.reset2) }
    })
  }
  property("object Module.currentModule should return an Option reference to the current Module") {
    def checkModule(mod: Module): Boolean = Module.currentModule.map(_ eq mod).getOrElse(false)
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle { })
      assert(Module.currentModule.get eq this)
      assert(checkModule(this))
    })
  }
  property("DataMirror.modulePorts should work") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle { })
      val m = Module(new chisel3.MultiIOModule {
        val a = IO(UInt(8.W))
        val b = IO(Bool())
      })
      assert(DataMirror.modulePorts(m) == Seq(
          "clock" -> m.clock, "reset" -> m.reset,
          "a" -> m.a, "b" -> m.b))
    })
  }
  property("A desiredName parameterized by a submodule should work") {
    ChiselStage.elaborate(new ModuleWrapper(new ModuleWire)).name should be ("ModuleWireWrapper")
  }
  property("A name generating a null pointer exception should provide a good error message") {
    (the [ChiselException] thrownBy extractCause[ChiselException] (ChiselStage.elaborate(new NullModuleWrapper)))
      .getMessage should include ("desiredName of chiselTests.NullModuleWrapper is null")
  }
  property("The name of a module in a function should be sane") {
    def foo = {
      class Foo1 extends RawModule {
        assert(name == "Foo1")
      }
      new Foo1
    }
    ChiselStage.elaborate(foo)
  }
  property("The name of an anonymous module should include '_Anon'") {
    trait Foo { this: RawModule =>
      assert(name.contains("_Anon"))
    }
    ChiselStage.elaborate(new RawModule with Foo)
  }
}
