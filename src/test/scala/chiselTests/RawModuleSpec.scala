// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{RawModule, withClockAndReset}
import chisel3.testers.BasicTester

class UnclockedPlusOne extends RawModule {
  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))

  out := in + 1.asUInt
}

class RawModuleTester extends BasicTester {
  val plusModule = Module(new UnclockedPlusOne)
  plusModule.in := 42.U
  assert(plusModule.out === 43.U)
  stop()
}

class PlusOneModule extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(32.W))
    val out = Output(UInt(32.W))
  })
  io.out := io.in + 1.asUInt
}

class RawModuleWithImplicitModule extends RawModule {
  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  val clk = IO(Input(Clock()))
  val rst = IO(Input(Bool()))

  withClockAndReset(clk, rst) {
    val plusModule = Module(new PlusOneModule)
    plusModule.io.in := in
    out := plusModule.io.out
  }
}

class ImplicitModuleInRawModuleTester extends BasicTester {
  val plusModule = Module(new RawModuleWithImplicitModule)
  plusModule.clk := clock
  plusModule.rst := reset
  plusModule.in := 42.U
  assert(plusModule.out === 43.U)
  stop()
}

class RawModuleWithDirectImplicitModule extends RawModule {
  val plusModule = Module(new PlusOneModule)
}

class ImplicitModuleDirectlyInRawModuleTester extends BasicTester {
  val plusModule = Module(new RawModuleWithDirectImplicitModule)
  stop()
}

class RawModuleSpec extends ChiselFlatSpec {
  "RawModule" should "elaborate" in {
    elaborate { new RawModuleWithImplicitModule }
  }

  "RawModule" should "work" in {
    assertTesterPasses({ new RawModuleTester })
  }

  "ImplicitModule in a withClock block in a RawModule" should "work" in {
    assertTesterPasses({ new ImplicitModuleInRawModuleTester })
  }


  "ImplicitModule directly in a RawModule" should "fail" in {
    intercept[chisel3.internal.ChiselException] {
      elaborate { new RawModuleWithDirectImplicitModule }
    }
  }

  "ImplicitModule directly in a RawModule in an ImplicitModule" should "fail" in {
    intercept[chisel3.internal.ChiselException] {
      elaborate { new ImplicitModuleDirectlyInRawModuleTester }
    }
  }
}
