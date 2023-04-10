package chiselTests

import chisel3._
import circt.stage.ChiselStage

class ProbeSpec extends ChiselFlatSpec with Utils {
  // Strip SourceInfos and split into lines
  private def processChirrtl(chirrtl: String): Array[String] =
    chirrtl.split('\n').map(line => line.takeWhile(_ != '@').trim())

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

    (processChirrtl(chirrtl) should contain).allOf(
      "output io : { flip in : Probe<UInt<1>>, out : Probe<UInt<1>>}",
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

}
