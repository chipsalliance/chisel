package chiselTests

import chisel3._
import circt.stage.ChiselStage

class ProbeSpec extends ChiselFlatSpec with Utils {
  // Strip SourceInfos and split into lines
  private def processChirrtl(chirrtl: String): Array[String] =
    chirrtl.split('\n').map(line => line.takeWhile(_ != '@').trim())

  "Simple probe usage" should "work" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new Module {
        val a = IO(Output(RWProbe(Bool())))

        val w = WireInit(Bool(), false.B)
        val w_probe = RWProbeValue(w)
        Probe.define(a, w_probe)
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

        Probe.define(u1.io.in, n)
        Probe.define(u2.io.in, u1.io.out)

        io.y := Probe.read(u2.io.out)

        Probe.forceInitial(u1.io.out, false.B)

        Probe.releaseInitial(u1.io.out)

        Probe.force(clock, io.x, u2.io.out, u1.io.out)
        Probe.release(clock, io.y, u2.io.out)
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

  "Connectors" should "work with probes" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new Module {

        class FooBundle extends Bundle {
          val bar = Bool()
          val baz = Probe(UInt(8.W)) // FIXME not supported in FIRRTL, maybe want to flatten probes in Chisel?
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

    (processChirrtl(chirrtl) should contain).allOf(
      "io.w.baz <= probe(io.a).baz",
      "io.w.bar <= probe(io.a).bar",
      "io.x.baz <= io.a.baz",
      "io.x.bar <= io.a.bar",
      "io.z.baz <= io.a.baz",
      "io.z.bar <= io.a.bar",
      "io.y.baz <= io.a.baz",
      "io.y.bar <= io.a.bar"
    )
  }

  "Probes" should "be able to access vector subindices" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new Module {

        class VecChild() extends RawModule {
          val io = IO(new Bundle {
            val in = Input(Vec(2, UInt(16.W)))
            val out = Output((Vec(2, Probe(Vec(2, UInt(16.W))))))
          })
          io.out.foreach { ele: Vec[UInt] => ele := io.in }
        }

        val io = IO(new Bundle {
          val out = Output(RWProbe(UInt(16.W)))
          val probeVec = Output(Probe(Vec(2, UInt(16.W))))
          val vecProbe = Output(Vec(2, Probe(UInt(16.W))))
        })

        val child = Module(new VecChild())

        // TODO how to handle define fowarding when taking elements from a vec

        // Probe.define(io.out, child.io.out(0)(1))
        // Probe.define(io.probeVec, Probe.probe(child.io.in))
        // io.vecProbe.foreach { vp =>
        //   Probe.define(vp, child.io.out)
        // }
      },
      Array("--full-stacktrace")
    )

    println(chirrtl)

    // (processChirrtl(chirrtl) should contain).allOf(
    //   "output io : { flip in : UInt<16>[2], out : RWProbe<UInt<16>>[2][2]}",
    //   "define io.out = child.io.out[0][1]"
    // )
  }

  "Wire() of a probe" should "fail" in {
    val exc = intercept[chisel3.ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
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
        new Module {
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
        new Module {
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
        new Module {
          val mem = SyncReadMem(1024, RWProbe(Vec(4, UInt(32.W))))
        },
        Array("--throw-on-first-error")
      )
    }
    exc.getMessage should be("Cannot make a Mem of a Chisel type with a probe modifier.")
  }

  // TODO probes of const types -- const of probe type?
}
