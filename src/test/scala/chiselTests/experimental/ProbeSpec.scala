package chiselTests.experimental

import chisel3._
import chisel3.experimental.Probe
import chiselTests.{ChiselFlatSpec, Utils}
import circt.stage.ChiselStage

class ProbeSpec extends ChiselFlatSpec with Utils {
  "Ref" should "emit FIRRTL probe descriptors" in {
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

}
