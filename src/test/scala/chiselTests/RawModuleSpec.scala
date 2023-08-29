// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.hierarchy.Definition
import chisel3.reflect.DataMirror
import chisel3.testers.BasicTester
import circt.stage.ChiselStage

class UnclockedPlusOne extends RawModule {
  val in = IO(Input(UInt(32.W)))
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
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
  })
  io.out := io.in + 1.asUInt
}

class RawModuleWithImplicitModule extends RawModule {
  val in = IO(Input(UInt(32.W)))
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

class RawModuleSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {
  "RawModule" should "elaborate" in {
    ChiselStage.emitCHIRRTL { new RawModuleWithImplicitModule }
  }

  "RawModule" should "work" in {
    assertTesterPasses({ new RawModuleTester })
  }

  "RawModule with atModuleBodyEnd" should "support late stage generators" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      atModuleBodyEnd {
        val extraPort0 = IO(Output(Bool()))
        extraPort0 := 0.B
      }

      atModuleBodyEnd {
        val extraPort1 = IO(Output(Bool()))
        extraPort1 := 1.B
      }
    })

    matchesAndOmits(chirrtl)(
      "output extraPort0 : UInt<1>",
      "output extraPort1 : UInt<1>",
      "connect extraPort0, UInt<1>(0h0)",
      "connect extraPort1, UInt<1>(0h1)"
    )()
  }

  "RawModule with atModuleBodyEnd" should "support multiple connects" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val port = IO(Output(UInt(2.W)))

      atModuleBodyEnd {
        port := 2.U
      }

      atModuleBodyEnd {
        port := 3.U
      }

      port := 1.U
    })

    matchesAndOmits(chirrtl)(
      "output port : UInt<2>",
      "connect port, UInt<1>(0h1)",
      "connect port, UInt<2>(0h2)",
      "connect port, UInt<2>(0h3)"
    )()
  }

  "RawModule with atModuleBodyEnd" should "support the added hardware in DataMirror" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val module = Module(new RawModule {
        val port0 = IO(Output(Bool()))
        port0 := 0.B

        atModuleBodyEnd {
          val port1 = IO(Output(Bool()))
          port1 := 0.B
        }
      })

      val mirroredPorts = DataMirror.modulePorts(module)

      mirroredPorts should have size 2
    })
  }

  "RawModule with atModuleBodyEnd" should "support the added hardware in Definition" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val definition = Definition(new RawModule {
        val port0 = IO(Output(Bool()))
        port0 := 0.B

        atModuleBodyEnd {
          val port1 = IO(Output(Bool()))
          port1 := 0.B
        }
      })

      val definitionPorts = Select.ios(definition)

      definitionPorts should have size 2
    })
  }

  "ImplicitModule in a withClock block in a RawModule" should "work" in {
    assertTesterPasses({ new ImplicitModuleInRawModuleTester })
  }

  "ImplicitModule directly in a RawModule" should "fail" in {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new RawModuleWithDirectImplicitModule }
      }
    }
  }

  "ImplicitModule directly in a RawModule in an ImplicitModule" should "fail" in {
    intercept[ChiselException] {
      extractCause[ChiselException] {
        ChiselStage.emitCHIRRTL { new ImplicitModuleDirectlyInRawModuleTester }
      }
    }
  }
}
