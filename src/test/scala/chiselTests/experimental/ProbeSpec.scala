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
    println(chirrtl)
    // chirrtl should include ("wire foo : const UInt<8>")
    // chirrtl should include ("reg bar : const SInt<4>")
  }

}
