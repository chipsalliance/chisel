// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.util.{Pipe, Valid}
import chiselTests.ChiselFlatSpec
import _root_.circt.stage.ChiselStage.emitCHIRRTL

class PipeSpec extends ChiselFlatSpec {
  behavior.of("Pipe")

  it should "Have decent names for Pipe(2)" in {
    class MyModule extends Module {
      val foo = IO(Input(Valid(UInt(8.W))))
      val bar = IO(Output(Valid(UInt(8.W))))
      bar := Pipe(foo.valid, bar.bits, 2)
    }
    val chirrtl = emitCHIRRTL(new MyModule)
    chirrtl should include("regreset bar_pipe_v")
    chirrtl should include("regreset bar_pipe_pipe_v")
    chirrtl should include("wire bar_pipe_pipe_out")
    chirrtl should include("connect bar_pipe_pipe_out.valid, bar_pipe_pipe_v")
    chirrtl should include("connect bar, bar_pipe_pipe_out")
  }

  it should "Have decent names for Pipe(0)" in {
    class MyModule extends Module {
      val foo = IO(Input(Valid(UInt(8.W))))
      val bar = IO(Output(Valid(UInt(8.W))))
      bar := Pipe(foo.valid, foo.bits, 0)
    }
    val chirrtl = emitCHIRRTL(new MyModule)
    (chirrtl should not).include("pipe")
    chirrtl should include("wire bar_out")
    chirrtl should include("connect bar_out.valid, foo.valid")
    chirrtl should include("connect bar, bar_out")
  }
}
