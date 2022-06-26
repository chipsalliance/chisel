package chiselTests.util

// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util.{Pipe, Valid}
import chisel3.stage.ChiselStage.emitChirrtl
import chisel3.experimental.FlatIO
import chiselTests.ChiselFlatSpec

class PipeSpec extends ChiselFlatSpec {
  behavior.of("Pipe")

  it should "Have decent names for Pipe(2)" in {
    class MyModule extends Module {
      val foo = IO(Input(Valid(UInt(8.W))))
      val bar = IO(Output(Valid(UInt(8.W))))
      bar := Pipe(foo.valid, bar.bits, 2)
    }
    val chirrtl = emitChirrtl(new MyModule)
    chirrtl should include("reg bar_pipe_v")
    chirrtl should include("reg bar_pipe_pipe_v")
    chirrtl should include("wire bar_pipe_pipe_out")
    chirrtl should include("bar_pipe_pipe_out.valid <= bar_pipe_pipe_v")
    chirrtl should include("bar <= bar_pipe_pipe_out")
  }

  it should "Have decent names for Pipe(0)" in {
    class MyModule extends Module {
      val foo = IO(Input(Valid(UInt(8.W))))
      val bar = IO(Output(Valid(UInt(8.W))))
      bar := Pipe(foo.valid, foo.bits, 0)
    }
    val chirrtl = emitChirrtl(new MyModule)
    (chirrtl should not).include("pipe")
    chirrtl should include("wire bar_out")
    chirrtl should include("bar_out.valid <= foo.valid")
    chirrtl should include("bar <= bar_out")
  }
}
