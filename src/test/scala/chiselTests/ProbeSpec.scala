// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe._
import chisel3.util.Counter
import chisel3.testers.{BasicTester, TesterDriver}
import circt.stage.ChiselStage

class ProbeSpec extends ChiselFlatSpec with Utils {
  // Strip SourceInfos and split into lines
  private def processChirrtl(chirrtl: String): Array[String] =
    chirrtl.split('\n').map(line => line.takeWhile(_ != '@').trim())

  "Simple probe usage" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val a = IO(Output(RWProbe(Bool())))

        val w = WireInit(Bool(), false.B)
        val w_probe = RWProbeValue(w)
        define(a, w_probe)
      },
      Array("--full-stacktrace")
    )
    processChirrtl(chirrtl) should contain("define a = rwprobe(w)")
  }

  "U-Turn example" should "emit FIRRTL probe statements and expressions" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new Module {

        val io = IO(new Bundle {
          val x = Input(Bool())
          val y = Output(Bool())
        })

        class UTurn() extends RawModule {
          val io = IO(new Bundle {
            val in = Input(RWProbe(Bool()))
            val out = Output(RWProbe(Bool()))
          })
          define(io.out, io.in)
        }

        val u1 = Module(new UTurn())
        val u2 = Module(new UTurn())

        val n = RWProbeValue(io.x)

        define(u1.io.in, n)
        define(u2.io.in, u1.io.out)

        io.y := read(u2.io.out)

        forceInitial(u1.io.out, false.B)

        releaseInitial(u1.io.out)

        when(io.x) { force(u2.io.out, u1.io.out) }
        when(io.y) { release(u2.io.out) }
      },
      Array("--full-stacktrace")
    )
    (processChirrtl(chirrtl) should contain).allOf(
      "output io : { flip in : RWProbe<UInt<1>>, out : RWProbe<UInt<1>>}",
      "define u1.io.in = rwprobe(io.x)",
      "define u2.io.in = u1.io.out",
      "connect io.y, read(u2.io.out)",
      "force_initial(u1.io.out, UInt<1>(0h0))",
      "release_initial(u1.io.out)",
      "force(clock, _T, u2.io.out, u1.io.out)",
      "release(clock, _T_1, u2.io.out)"
    )
  }

  "Probe methods in when contexts" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val in = IO(Input(Bool()))
        val out = IO(Output(RWProbe(Bool())))

        val w = WireInit(Bool(), false.B)

        when(in) {
          define(out, RWProbeValue(in))
          w := read(out)
        }
      },
      Array("--full-stacktrace")
    )
    (processChirrtl(chirrtl) should contain).allOf(
      "when in :",
      "define out = rwprobe(in)",
      "connect w, read(out)"
    )
  }

  "Subfields of probed bundles" should "be accessible" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {

        class FooBundle() extends Bundle {
          val a = Bool()
          val b = Bool()
        }

        class Foo() extends RawModule {
          val p = IO(Output(Probe(new FooBundle())))
          val x = Wire(new FooBundle())
          define(p, ProbeValue(x))
        }

        val x = IO(Output(Bool()))
        val y = IO(Output(Probe(Bool())))
        val f = Module(new Foo())
        x := read(f.p.b)
        define(y, f.p.b)
      },
      Array("--full-stacktrace")
    )

    (processChirrtl(chirrtl) should contain).allOf(
      "output p : Probe<{ a : UInt<1>, b : UInt<1>}>",
      "wire x : { a : UInt<1>, b : UInt<1>}",
      "define p = probe(x)",
      "connect x, read(f.p.b)",
      "define y = f.p.b"
    )
  }

  "Subindices of probed vectors" should "be accessible" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        class VecChild() extends RawModule {
          val p = IO(Output(Vec(2, Probe(Vec(2, UInt(16.W))))))
        }

        val outProbe = IO(Output(Probe(UInt(16.W))))
        val child = Module(new VecChild())
        define(outProbe, child.p(0)(1))
      },
      Array("--full-stacktrace")
    )

    (processChirrtl(chirrtl) should contain).allOf(
      "output p : Probe<UInt<16>[2]>[2]",
      "output outProbe : Probe<UInt<16>>",
      "define outProbe = child.p[0][1]"
    )
  }

  "Properly excluded bulk connectors" should "work with Bundles containing probes" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        class FooBundle() extends Bundle {
          val bar = Probe(Bool())
          val baz = UInt(4.W)
          val qux = Flipped(Probe(UInt(4.W)))
          val fizz = Flipped(Bool())
        }
        val io = IO(new Bundle {
          val in = Flipped(new FooBundle())
          val a = new FooBundle()
          val b = new FooBundle()
          val c = new FooBundle()
          val d = Output(new FooBundle())
        })

        io.a.exclude(_.bar, _.qux) :<>= io.in.exclude(_.bar, _.qux)
        io.b.exclude(_.bar, _.qux) :<= io.in.exclude(_.bar, _.qux)
        io.c.excludeProbes :>= io.in.excludeProbes
        io.d.excludeProbes :#= io.in.excludeProbes
      },
      Array("--full-stacktrace")
    )

    (processChirrtl(chirrtl) should contain).allOf(
      "connect io.in.fizz, io.a.fizz",
      "connect io.a.baz, io.in.baz",
      "connect io.b.baz, io.in.baz",
      "connect io.in.fizz, io.c.fizz",
      "connect io.d.fizz, io.in.fizz",
      "connect io.d.baz, io.in.baz"
    )
  }

  ":<>= connector" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        class FooBundle extends Bundle {
          val bar = Probe(Bool())
          val baz = UInt(4.W)
        }
        val io = IO(new Bundle {
          val in = Input(new FooBundle)
          val out = Output(new FooBundle)
        })
        io.out :<>= io.in
      }
    )
    processChirrtl(chirrtl) should contain("define io.out.bar = io.in.bar")
    processChirrtl(chirrtl) should contain("connect io.out.baz, io.in.baz")
  }

  ":<>= connector with probes of aggregates" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        class FooBundle extends Bundle {
          val baz = UInt(4.W)
        }
        val io = IO(new Bundle {
          val in = Input(Probe(new FooBundle))
          val out = Output(Probe(new FooBundle))
        })
        io.out :<>= io.in
      }
    )
    processChirrtl(chirrtl) should contain("define io.out = io.in")
    processChirrtl(chirrtl) should not contain ("out.baz")
  }

  "Mismatched probe/non-probe with :<>= connector" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val io = IO(new Bundle {
            val in = Input(Vec(2, Bool()))
            val out = Output(Vec(2, Probe(Bool())))
          })
          io.out :<>= io.in
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include(
      "mismatched probe/non-probe types in ProbeSpec_Anon.io.out[0]: IO[Bool] and ProbeSpec_Anon.io.in[0]: IO[Bool]."
    )
  }

  ":= connector with probe/non-probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val io = IO(new Bundle {
            val in = Input(Bool())
            val out = Output(Probe(Bool()))
          })
          io.out := io.in
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include(
      "Connection between sink (ProbeSpec_Anon.io.out: IO[Bool]) and source (ProbeSpec_Anon.io.in: IO[Bool]) failed @: Sink io.out in ProbeSpec_Anon of Probed type cannot participate in a mono connection (:=)"
    )
  }

  ":= connector with probe" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val io = IO(new Bundle {
          val in = Input(Probe(Bool()))
          val out = Output(Probe(Bool()))
        })
        io.out := io.in
      }
    )
    processChirrtl(chirrtl) should contain("define io.out = io.in")
  }

  ":= connector with probes but in wrong direction" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      val chirrtl = ChiselStage.emitCHIRRTL(
        new RawModule {
          val io = IO(new Bundle {
            val in = Input(Probe(Bool()))
            val out = Output(Probe(Bool()))
          })
          io.in := io.out
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include(
      "Connection between sink (ProbeSpec_Anon.io.in: IO[Bool]) and source (ProbeSpec_Anon.io.out: IO[Bool]) failed @: io.in in ProbeSpec_Anon cannot be written from module ProbeSpec_Anon"
    )
  }

  ":= connector with aggregate of probe/non-probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val io = IO(new Bundle {
            val in = Input(Vec(2, Bool()))
            val out = Output(Vec(2, Probe(Bool())))
          })
          io.out := io.in
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include(
      "Connection between sink (ProbeSpec_Anon.io.out: IO[Bool[2]]) and source (ProbeSpec_Anon.io.in: IO[Bool[2]]) failed @: (0)Sink io.out[0] in ProbeSpec_Anon of Probed type cannot participate in a mono connection (:=)"
    )
  }

  "<> connector" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val io = IO(new Bundle {
            val in = Input(Vec(2, Probe(Bool())))
            val out = Output(Vec(2, Probe(Bool())))
          })
          io.out <> io.in
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should be(
      "Connection between left (ProbeSpec_Anon.io.out: IO[Bool[2]]) and source (ProbeSpec_Anon.io.in: IO[Bool[2]]) failed @Left of Probed type cannot participate in a bi connection (<>)"
    )
  }

  ":= connector with aggregates of probe" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val io = IO(new Bundle {
          val in = Input(Vec(2, Probe(Bool())))
          val out = Output(Vec(2, Probe(Bool())))
        })
        io.out := io.in
      }
    )
    processChirrtl(chirrtl) should contain("define io.out[0] = io.in[0]")
    processChirrtl(chirrtl) should contain("define io.out[1] = io.in[1]")
  }

  "Probe define between non-connectable data types" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val p = IO(Output(Probe(UInt(4.W))))

          val w = WireInit(Bool(), false.B)
          val v = ProbeValue(w)

          define(p, v)
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot define a probe on a non-equivalent type.")
  }

  "Probe of a probe type" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val a = Output(RWProbe(Probe(Bool())))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot probe a probe.")
  }

  "Probes of aggregates containing probes" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val out = IO(Output(Probe(new Bundle {
            val a = Probe(Bool())
          })))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot create a probe of an aggregate containing a probe.")
  }

  "Wire() of a probe" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(Probe(Bool()))
    })
    processChirrtl(chirrtl) should contain("wire w : Probe<UInt<1>>")
  }

  "WireInit of a probe" should "work" in {
    class Test extends RawModule {
      val init = WireInit(Bool(), false.B)
      val w = WireInit(RWProbe(Bool()), RWProbeValue(init))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new Test)
    processChirrtl(chirrtl) should contain("wire w : RWProbe<UInt<1>>")
    processChirrtl(chirrtl) should contain("define w = rwprobe(init)")
    ChiselStage.emitSystemVerilog(new Test)
  }

  "Reg() of a probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val w = Reg(RWProbe(Bool()))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot make a register of a Chisel type with a probe modifier.")
  }

  "RegInit of a probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val w = RegInit(Probe(Bool()), false.B)
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot make a register of a Chisel type with a probe modifier.")
  }

  "Memories of probes" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val mem = SyncReadMem(1024, RWProbe(Vec(4, UInt(32.W))))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot make a Mem of a Chisel type with a probe modifier.")
  }

  "Defining a Probe with a rwprobe()" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val in = IO(Input(Bool()))
        val out = IO(Output(Probe(Bool())))
        define(out, RWProbeValue(in))
      },
      Array("--full-stacktrace")
    )
    processChirrtl(chirrtl) should contain("define out = rwprobe(in)")
  }

  "Defining a RWProbe with a probe()" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val in = IO(Input(Bool()))
          val out = IO(Output(RWProbe(Bool())))
          define(out, ProbeValue(in))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot use a non-writable probe expression to define a writable probe.")
  }

  "Force of a non-writable Probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val in = IO(Input(Bool()))
          val out = IO(Output(Probe(Bool())))
          force(out, in)
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot force a non-writable Probe.")
  }

  "RWProbeValue() on a literal" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val out = IO(Output(RWProbe(Bool())))
          define(out, RWProbeValue(true.B))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot get a probe value from a literal.")
  }

  "ProbeValue() on a literal" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val out = IO(Output(Probe(Bool())))
        define(out, ProbeValue(true.B))
      },
      Array("--full-stacktrace")
    )
    (processChirrtl(chirrtl) should contain).allOf(
      "output out : Probe<UInt<1>>",
      "wire lit_probe_val : UInt<1>",
      "connect lit_probe_val, UInt<1>(0h1)",
      "define out = probe(lit_probe_val)"
    )
  }

  "Probes of Const type" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val out = IO(Probe(Const(Bool())))
      },
      Array("--full-stacktrace")
    )
    processChirrtl(chirrtl) should contain("output out : Probe<const UInt<1>>")
  }

  "RWProbes of Const type" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val out = IO(RWProbe(Const(Bool())))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Cannot create a writable probe of a const type.")
  }

  "Probe force methods" should "properly extend values that are not wide enough" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new Module {
        val in = IO(Input(UInt(4.W)))
        val p = IO(Output(RWProbe(UInt(16.W))))
        forceInitial(p, 123.U)
        force(p, in)
      },
      Array("--full-stacktrace")
    )
    (processChirrtl(chirrtl) should contain).allOf(
      "node _T = pad(UInt<7>(0h7b), 16)",
      "force_initial(p, _T)",
      "node _T_2 = pad(in, 16)",
      "force(clock, _T_1, p, _T_2)"
    )
  }

  it should "error out with constants that are too wide" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val a = IO(Output(RWProbe(UInt(2.W))))
          forceInitial(a, 123.U)
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Data width 7 is larger than 2.")
  }

  it should "error out on Wires of unknown widths" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val in = IO(Input(UInt()))
          val p = IO(Output(RWProbe(UInt(16.W))))
          force(p, in)
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Data width unknown.")
  }

  it should "error out on probes of unknown widths" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val in = IO(Input(UInt(16.W)))
          val p = IO(Output(RWProbe(UInt())))
          force(p, in)
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should include("Probe width unknown.")
  }

  "Probe force/release reg example" should "work in simulator" in {
    // Simple example forcing a register and checking basic behavior.

    // Demonstrate that a bundle with data + probes works.
    class MiniBundle extends Bundle {
      val x = Flipped(UInt(16.W))
      val refs = new Bundle {
        val out = RWProbe(UInt(16.W))
        val reg = RWProbe(UInt(16.W))
      }
    }
    class Top extends Module {
      val b = IO(new MiniBundle)
      val out = IO(Output(UInt(16.W)))

      val r = Reg(UInt(16.W))
      r := b.x
      out := r

      // Export rwprobe's to various signals.
      // Verilator errors if we attempt to force an input port,
      // so don't include that in this test.
      define(b.refs.out, RWProbeValue(out))
      define(b.refs.reg, RWProbeValue(r))
    }
    runTester(new BasicTester {
      val dut = Module(new Top)

      val (cycle, done) = Counter(true.B, 20)
      dut.b.x := 42.U

      chisel3.assert(dut.b.x === 42.U)
      chisel3.assert(read(dut.b.refs.out) === dut.out)

      // Force cycle to register.
      // Verilator's force (as documented) doesn't update when the RHS changes
      // which it should per SV spec.  Workaround by forcing repeatedly below.
      forceInitial(dut.b.refs.reg, cycle)
      // Additionally, 'initial force ...' doesn't seem to work here (?).
      // So do this on cycle zero explicitly for compatibility.
      when(cycle === 0.U) { force(dut.b.refs.reg, cycle) }

      when(0.U < cycle && cycle <= 10.U) {
        // Force cycle and check we observe it on output a cycle later.
        chisel3.assert(dut.out === cycle - 1.U)
        force(dut.b.refs.reg, cycle)
      }.elsewhen(cycle === 11.U) {
        // Check last value, release.
        chisel3.assert(dut.out === 10.U)
        release(dut.b.refs.reg)
      }.elsewhen(cycle === 12.U) {
        // Check original value is restored.
        chisel3.assert(dut.out === 42.U)
      }
      // Force the register and the output port.
      when(cycle >= 13.U) {
        force(dut.b.refs.reg, cycle)
        force(dut.b.refs.out, 123.U)
      }
      when(cycle > 13.U) {
        // Register reads the value forced to it.
        chisel3.assert(read(dut.b.refs.reg) === cycle)
        // Output signal should have value forced to it.
        chisel3.assert(dut.out === 123.U)
      }

      when(done) { stop() }
    }) should be(true)
  }

  "Enum probe" should "work" in {
    object MyEnum extends ChiselEnum {
      val e0, e1, e2 = Value
    }
    class TestMod extends RawModule {
      val a = IO(Output(RWProbe(MyEnum())))

      val w = WireInit(MyEnum(), MyEnum.e1)
      val w_probe = RWProbeValue(w)
      define(a, w_probe)
    }
    ChiselStage.emitSystemVerilog(new TestMod)
  }
}
