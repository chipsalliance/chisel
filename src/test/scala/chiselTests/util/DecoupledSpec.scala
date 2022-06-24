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
   input in : { valid : UInt<1>, bits : UInt<8>}
   output out : { valid : UInt<1>, bits : UInt<8>}

   reg out_v : UInt<1>, clock with :
     reset => (reset, UInt<1>("h0")) @[Valid.scala 129:43]
   out_v <= in.valid @[Valid.scala 129:43]
   reg out_b : UInt<8>, clock with :
     reset => (UInt<1>("h0"), out_b) @[Valid.scala 130:44]
   when in.valid : @[Valid.scala 130:44]
     out_b <= in.bits @[Valid.scala 130:44]
   reg out_out_v : UInt<1>, clock with :
     reset => (reset, UInt<1>("h0")) @[Valid.scala 129:43]
   out_out_v <= out_v @[Valid.scala 129:43]
   reg out_out_b : UInt<8>, clock with :
     reset => (UInt<1>("h0"), out_out_b) @[Valid.scala 130:44]
   when out_v : @[Valid.scala 130:44]
     out_out_b <= out_b @[Valid.scala 130:44]
   wire out_out : { valid : UInt<1>, bits : UInt<8>} @[Valid.scala 124:21]
   out_out.valid <= out_out_v @[Valid.scala 125:17]
   out_out.bits <= out_out_b @[Valid.scala 126:16]
   out <= out_out @[DecoupledSpec.scala 48:11]


*/

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
