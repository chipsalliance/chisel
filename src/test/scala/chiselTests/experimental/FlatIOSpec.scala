// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.util.Valid
import chisel3.stage.ChiselStage.emitChirrtl
import chisel3.experimental.FlatIO
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
    val chirrtl = emitChirrtl(new MyModule)
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
    val chirrtl = emitChirrtl(new MyModule)
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
    val chirrtl = emitChirrtl(new MyModule)
    chirrtl should include("out[addr] <= in[addr]")
  }
}
