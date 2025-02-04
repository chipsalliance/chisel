// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.aop.Select
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
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

class RawModuleSpec extends ChiselFlatSpec with Utils with FileCheck {
  "RawModule" should "elaborate" in {
    ChiselStage.emitCHIRRTL { new RawModuleWithImplicitModule }
  }

  "RawModule" should "work" in {
    assertTesterPasses({ new RawModuleTester })
  }

  "RawModule with atModuleBodyEnd" should "support late stage generators" in {
    generateFirrtlAndFileCheck {
      new RawModule {
        atModuleBodyEnd {
          val extraPort0 = IO(Output(Bool()))
          extraPort0 := 0.B
        }

        atModuleBodyEnd {
          val extraPort1 = IO(Output(Bool()))
          extraPort1 := 1.B
        }
      }
    }(
      """|CHECK-LABEL: public module
         |CHECK:         output extraPort0 : UInt<1>
         |CHECK:         output extraPort1 : UInt<1>
         |CHECK:         connect extraPort0, UInt<1>(0h0)
         |CHECK:         connect extraPort1, UInt<1>(0h1)
         |""".stripMargin
    )
  }

  "RawModule with atModuleBodyEnd" should "support multiple connects" in {
    generateFirrtlAndFileCheck {
      new RawModule {
        val port = IO(Output(UInt(2.W)))

        atModuleBodyEnd {
          port := 2.U
        }

        atModuleBodyEnd {
          port := 3.U
        }

        port := 1.U
      }
    }(
      """|CHECK-LABEL: public module
         |CHECK:         output port : UInt<2>
         |CHECK:         connect port, UInt<1>(0h1)
         |CHECK:         connect port, UInt<2>(0h2)
         |CHECK:         connect port, UInt<2>(0h3)
         |""".stripMargin
    )
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

  "RawModule with afterModuleBuilt" should "be able to create other modules" in {
    generateFirrtlAndFileCheck {
      new RawModule {
        override def desiredName = "Foo"
        val port0 = IO(Input(Bool()))

        afterModuleBuilt {
          Module(new RawModule {
            override def desiredName = "Bar"
            val port1 = IO(Input(Bool()))
          })
        }
      }
    }(
      """|CHECK-LABEL: module Foo :
         |CHECK:         input port0 : UInt<1>
         |CHECK-LABEL: public module Bar :
         |CHECK:         input port1 : UInt<1>
         |""".stripMargin
    )
  }

  "RawModule with afterModuleBuilt" should "be able to instantiate the surrounding module" in {
    @instantiable
    class Foo extends RawModule {
      override def desiredName = "Foo"
      @public val port0 = IO(Input(Bool()))

      afterModuleBuilt {
        val fooDef = this.toDefinition
        Module(new RawModule {
          override def desiredName = "Bar"
          val port1 = IO(Input(Bool()))
          val foo1 = Instance(fooDef)
          val foo2 = Instance(fooDef)
          foo1.port0 := port1
          foo2.port0 := port1
        })
      }
    }

    generateFirrtlAndFileCheck(new Foo)(
      """|CHECK-LABEL: module Foo :
         |CHECK:         input port0 : UInt<1>
         |CHECK-LABEL: public module Bar
         |CHECK:         input port1 : UInt<1>
         |CHECK:         inst foo1 of Foo
         |CHECK:         inst foo2 of Foo
         |CHECK:         connect foo1.port0, port1
         |CHECK:         connect foo2.port0, port1
         |""".stripMargin
    )
  }

  "RawModule marked as formal test" should "emit a formal test declaration" in {
    class Foo extends RawModule {
      FormalTest(this)
      FormalTest(this, MapTestParam(Map("hello" -> StringTestParam("world"))))
      FormalTest(
        this,
        MapTestParam(
          Map(
            "a_int" -> IntTestParam(42),
            "b_double" -> DoubleTestParam(13.37),
            "c_string" -> StringTestParam("hello"),
            "d_array" -> ArrayTestParam(Seq(IntTestParam(42), StringTestParam("hello"))),
            "e_map" -> MapTestParam(
              Map(
                "x" -> IntTestParam(42),
                "y" -> StringTestParam("hello")
              )
            )
          )
        ),
        "thisBetterWork"
      )
    }

    generateFirrtlAndFileCheck(new Foo)(
      """|CHECK: formal Foo of [[FOO:Foo_.*]] :
         |CHECK: formal Foo_1 of [[FOO]] :
         |CHECK:   hello = "world"
         |CHECK: formal thisBetterWork of [[FOO]] :
         |CHECK:   a_int = 42
         |CHECK:   b_double = 13.37
         |CHECK:   c_string = "hello"
         |CHECK:   d_array = [42, "hello"]
         |CHECK:   e_map = {x = 42, y = "hello"}
         |CHECK: module [[FOO]] :
         |""".stripMargin
    )
  }
}
