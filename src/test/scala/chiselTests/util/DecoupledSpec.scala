package chiselTests.util

// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util.{Pipe, Valid}
import chisel3.stage.ChiselStage.emitChirrtl
import chisel3.experimental.FlatIO
import chiselTests.ChiselFlatSpec

class DecoupledSpec extends ChiselFlatSpec {
  behavior.of("Decoupled")

/* 
With TransitName:

module MyModule :
  input clock : Clock
  input reset : UInt<1>
  input in : { valid : UInt<1>, bits : UInt<8>}
  output out : { valid : UInt<1>, bits : UInt<8>}

  reg out_v : UInt<1>, clock with :
    reset => (reset, UInt<1>("h0")) @[Valid.scala 127:22]
  out_v <= in.valid @[Valid.scala 127:22]
  reg out_b : UInt<8>, clock with :
    reset => (UInt<1>("h0"), out_b) @[Valid.scala 128:24]
  when in.valid : @[Valid.scala 128:24]
    out_b <= in.bits @[Valid.scala 128:24]
  reg out_outPipe_valid : UInt<1>, clock with :
    reset => (reset, UInt<1>("h0")) @[Valid.scala 127:22]
  out_outPipe_valid <= out_v @[Valid.scala 127:22]
  reg out_outPipe_bits : UInt<8>, clock with :
    reset => (UInt<1>("h0"), out_outPipe_bits) @[Valid.scala 128:24]
  when out_v : @[Valid.scala 128:24]
    out_outPipe_bits <= out_b @[Valid.scala 128:24]
  wire out_out : { valid : UInt<1>, bits : UInt<8>} @[Valid.scala 122:21]
  out_out.valid <= out_outPipe_valid @[Valid.scala 123:17]
  out_out.bits <= out_outPipe_bits @[Valid.scala 124:16]
  out <= out_out @[DecoupledSpec.scala 18:11]

With prefix(...):

module MyModule :
     input clock : Clock
     input reset : UInt<1>
     input foo : { valid : UInt<1>, bits : UInt<8>}
     output bar : { valid : UInt<1>, bits : UInt<8>}
 
     reg bar_pipe_v : UInt<1>, clock with :
       reset => (reset, UInt<1>("h0")) @[Valid.scala 129:24]
     bar_pipe_v <= foo.valid @[Valid.scala 129:24]
     reg bar_pipe_b : UInt<8>, clock with :
       reset => (UInt<1>("h0"), bar_pipe_b) @[Valid.scala 130:26]
     when foo.valid : @[Valid.scala 130:26]
       bar_pipe_b <= bar.bits @[Valid.scala 130:26]
     reg bar_pipe_pipe_v : UInt<1>, clock with :
       reset => (reset, UInt<1>("h0")) @[Valid.scala 129:24]
     bar_pipe_pipe_v <= bar_pipe_v @[Valid.scala 129:24]
     reg bar_pipe_pipe_b : UInt<8>, clock with :
       reset => (UInt<1>("h0"), bar_pipe_pipe_b) @[Valid.scala 130:26]
     when bar_pipe_v : @[Valid.scala 130:26]
       bar_pipe_pipe_b <= bar_pipe_b @[Valid.scala 130:26]
     wire bar_pipe_pipe_out : { valid : UInt<1>, bits : UInt<8>} @[Valid.scala 124:21]
     bar_pipe_pipe_out.valid <= bar_pipe_pipe_v @[Valid.scala 125:17]
     bar_pipe_pipe_out.bits <= bar_pipe_pipe_b @[Valid.scala 126:16]
     bar <= bar_pipe_pipe_out @[DecoupledSpec.scala 77:11]
*/

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
    chirrtl should not include("pipe")
    chirrtl should include ("wire bar_out")
    chirrtl should include ("bar_out.valid <= foo.valid")
    chirrtl should include ("bar <= bar_out")
  }
}
