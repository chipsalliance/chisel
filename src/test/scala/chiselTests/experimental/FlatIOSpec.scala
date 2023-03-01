// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.util.Valid
import circt.stage.ChiselStage.emitCHIRRTL
import chisel3.experimental.{Analog, FlatIO}
import chiselTests.ChiselFlatSpec

class FlatIOSpec extends ChiselFlatSpec {
  behavior.of("FlatIO")

  it should "create ports without a prefix" in {
    class MyModule extends RawModule {
      val io = FlatIO(new Bundle {
        val in = Input(UInt(8.W))
        val out = Output(UInt(8.W))
      })
      io.out := io.in
    }
    val chirrtl = emitCHIRRTL(new MyModule)
    chirrtl should include("input in : UInt<8>")
    chirrtl should include("output out : UInt<8>")
    chirrtl should include("out <= in")
  }

  it should "support bulk connections between FlatIOs and regular IOs" in {
    class MyModule extends RawModule {
      val in = FlatIO(Input(Valid(UInt(8.W))))
      val out = IO(Output(Valid(UInt(8.W))))
      out := in
    }
    val chirrtl = emitCHIRRTL(new MyModule)
    chirrtl should include("out.bits <= bits")
    chirrtl should include("out.valid <= valid")
  }

  it should "dynamically indexing Vecs inside of FlatIOs" in {
    class MyModule extends RawModule {
      val io = FlatIO(new Bundle {
        val addr = Input(UInt(2.W))
        val in = Input(Vec(4, UInt(8.W)))
        val out = Output(Vec(4, UInt(8.W)))
      })
      io.out(io.addr) := io.in(io.addr)
    }
    val chirrtl = emitCHIRRTL(new MyModule)
    chirrtl should include("out[addr] <= in[addr]")
  }

  it should "support Analog members" in {
    class MyBundle extends Bundle {
      val foo = Output(UInt(8.W))
      val bar = Analog(8.W)
    }
    class MyModule extends RawModule {
      val io = FlatIO(new Bundle {
        val in = Flipped(new MyBundle)
        val out = new MyBundle
      })
      io.out <> io.in
    }
    val chirrtl = emitCHIRRTL(new MyModule)
    chirrtl should include("out.foo <= in.foo")
    chirrtl should include("attach (out.bar, in.bar)")
  }
}
