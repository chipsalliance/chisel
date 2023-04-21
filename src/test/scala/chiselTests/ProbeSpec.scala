// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe._
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
            val in = Input(Probe(Bool()))
            val out = Output(RWProbe(Bool()))
          })
          io.out := io.in
        }

        val u1 = Module(new UTurn())
        val u2 = Module(new UTurn())

        val n = ProbeValue(io.x)

        define(u1.io.in, n)
        define(u2.io.in, u1.io.out)

        io.y := read(u2.io.out)

        forceInitial(u1.io.out, false.B)

        releaseInitial(u1.io.out)

        force(clock, io.x, u2.io.out, u1.io.out)
        release(clock, io.y, u2.io.out)
      },
      Array("--full-stacktrace")
    )

    (processChirrtl(chirrtl) should contain).allOf(
      "output io : { flip in : Probe<UInt<1>>, out : RWProbe<UInt<1>>}",
      "define u1.io.in = probe(io.x)",
      "define u2.io.in = u1.io.out",
      "io.y <= read(u2.io.out)",
      "force_initial(u1.io.out, UInt<1>(\"h0\"))",
      "release_initial(u1.io.out)",
      "force(clock, io.x, u2.io.out, u1.io.out)",
      "release(clock, io.y, u2.io.out)"
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
          x := DontCare
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
      "x <= read(f.p.b)",
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

  // FIXME
  "Connectors" should "work with probes" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {

        class FooBundle extends Bundle {
          val bar = Bool()
          val baz = UInt(8.W)
        }

        val io = IO(new Bundle {
          val a = Input(new FooBundle)
          val w = Output(Probe(new FooBundle))
          val x = Output(new FooBundle)
          val y = Output(new FooBundle)
          val z = Output(new FooBundle)
        })

        // connecting two probe types
        io.w := ProbeValue(io.a) // FIXME this should be a define

        // connecting bundles containing probe types
        // FIXME these error -- talk to Adam
        io.x := io.a
        io.z <> io.a
        io.y :<>= io.a
      },
      Array("--full-stacktrace")
    )

    println(chirrtl)

    // (processChirrtl(chirrtl) should contain).allOf(
    //   "io.w.baz <= probe(io.a).baz",
    //   "io.w.bar <= probe(io.a).bar",
    //   "io.x.baz <= io.a.baz",
    //   "io.x.bar <= io.a.bar",
    //   "io.z.baz <= io.a.baz",
    //   "io.z.bar <= io.a.bar",
    //   "io.y.baz <= io.a.baz",
    //   "io.y.bar <= io.a.bar"
    // )
  }

  // TODO define with non-connectable src/sink should fail

  "Probe of a probe type" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val a = Output(RWProbe(Probe(Bool())))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should be("Cannot probe a probe.")
  }

  // TODO
  // "Probes of aggregates containing probes" should "fail" in {
  //   val chirrtl = ChiselStage.emitCHIRRTL(
  //     new Module {

  //       class FooBundle extends Bundle {
  //         val bar = Probe(UInt(8.W))
  //       }

  //       val foo = new FooBundle
  //       val probeFoo = ProbeValue(foo)
  //     },
  //     Array("--full-stacktrace")
  //   )

  //   println(chirrtl)
  //   // FIXME
  // }

  "Wire() of a probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val w = Wire(Probe(Bool()))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should be("Cannot make a wire of a Chisel type with a probe modifier.")
  }

  "WireInit of a probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new RawModule {
          val w = WireInit(RWProbe(Bool()), false.B)
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should be("Cannot make a wire of a Chisel type with a probe modifier.")
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
    exc.getMessage should be("Cannot make a register of a Chisel type with a probe modifier.")
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
    exc.getMessage should be("Cannot make a register of a Chisel type with a probe modifier.")
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
    exc.getMessage should be("Cannot make a Mem of a Chisel type with a probe modifier.")
  }

  // TODO probes of const type should work

  // TODO writable probes of const type should fail

  // TODO const of probe type should fail?
}
