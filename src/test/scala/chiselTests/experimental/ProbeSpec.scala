package chiselTests.experimental

import chisel3._
import chisel3.experimental.Probe
import chiselTests.{ChiselFlatSpec, Utils}
import circt.stage.ChiselStage

class ProbeSpec extends ChiselFlatSpec with Utils {
  "U-Turn example" should "emit FIRRTL probe statements and expressions" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new Module {

        val io = IO(new Bundle {
          val x = Input(UInt(1.W))
          val y = Output(UInt(1.W))
        })

        class UTurn() extends RawModule {
          val io = IO(new Bundle {
            val in = Input(Probe(Bool()))
            val out = Output(Probe(Bool()))
          })
          io.out := io.in
        }

        val u1 = Module(new UTurn())
        val u2 = Module(new UTurn())

        val n = Probe.probe(io.x)

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

    chirrtl should include("output io : { flip in : Probe<UInt<1>>, out : Probe<UInt<1>>}")
    chirrtl should include("define u1.io.in = probe(io.x)")
    chirrtl should include("define u2.io.in = u1.io.out")
    chirrtl should include("io.y <= read(u2.io.out)")
    chirrtl should include("force_initial(u1.io.out, UInt<1>(\"h0\")")
    chirrtl should include("release_initial(u1.io.out")
    chirrtl should include("force(clock, io.x, u2.io.out, u1.io.out)")
    chirrtl should include("release(clock, io.y, u2.io.out)")
  }

  "Connectors" should "work with probes" in {
    val chirrtl = ChiselStage.emitCHIRRTL(
      new Module {

        class FooBundle extends Bundle {
          val bar = Bool()
          val baz = Probe(UInt(8.W))
        }

        val io = IO(new Bundle {
          val a = Input(new FooBundle)
          val w = Output(Probe(new FooBundle))
          val x = Output(new FooBundle)
          val y = Output(new FooBundle)
          val z = Output(new FooBundle)
        })

        // connecting two probe types
        io.w := Probe.probe(io.a)

        // connecting bundles containing probe types
        io.x := io.a
        io.z <> io.a
        io.y :<>= io.a
      },
      Array("--full-stacktrace")
    )

    chirrtl should include("io.w.baz <= probe(io.a).baz")
    chirrtl should include("io.w.bar <= probe(io.a).bar")
    chirrtl should include("io.x.baz <= io.a.baz")
    chirrtl should include("io.x.bar <= io.a.bar")
    chirrtl should include("io.z.baz <= io.a.baz")
    chirrtl should include("io.z.bar <= io.a.bar")
    chirrtl should include("io.y.baz <= io.a.baz")
    chirrtl should include("io.y.bar <= io.a.bar")
  }

}
