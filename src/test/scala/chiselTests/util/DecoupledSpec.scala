package chiselTests.util

// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util.{Pipe, Valid}
import chisel3.stage.ChiselStage.emitChirrtl
import chisel3.experimental.FlatIO
import chiselTests.ChiselFlatSpec

class DecoupledSpec extends ChiselFlatSpec {
  behavior.of("Decoupled")

  it should "Have decent names for Pipe" in {
    class MyModule extends Module {
      val in = IO(Input(Valid(UInt(8.W))))
      val out = IO(Output(Valid(UInt(8.W))))
      out := Pipe(in.valid, in.bits, 3)
    }
    val chirrtl = emitChirrtl(new MyModule)
    chirrtl should include("out.bits <= bits")
    chirrtl should include("out.valid <= valid")
  }
}
