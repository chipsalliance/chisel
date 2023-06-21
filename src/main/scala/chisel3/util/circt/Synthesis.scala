// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.IntrinsicModule
import chisel3.internal.Builder

import circt.Intrinsic

/** A 2-to-1 mux cell intrinsic.
  */
private class MuxCell2Intrinsic[T <: Data](gen: T) extends IntrinsicModule("circt_muxcell2") {
  val sel = IO(Input(UInt(1.W)))
  val high = IO(Input(gen))
  val low = IO(Input(gen))
  val out = IO(Output(gen))
}

object MuxCell2 {

  /** Creates an intrinsic which is lowered to a 2-to-1 MUX cell in synthesis tools.
    * @example {{{
    * v := MuxCell2(sel, high, low)
    * }}}
    */

  def apply[T <: Data](sel: UInt, high: T, low: T): T = {
    val inst = Module(new MuxCell2Intrinsic(chisel3.chiselTypeOf(high)))
    inst.sel := sel
    inst.high := high
    inst.low := low
    inst.out
  }
}

/** A 4-to-1 mux cell intrinsic.
  */
private class MuxCell4Intrinsic[T <: Data](gen: T) extends IntrinsicModule("circt_muxcell4") {
  val sel = IO(Input(UInt(2.W)))
  val v3 = IO(Input(gen))
  val v2 = IO(Input(gen))
  val v1 = IO(Input(gen))
  val v0 = IO(Input(gen))
  val out = IO(Output(gen))
}

object MuxCell4 {

  /** Creates an intrinsic which is lowered to a 4-to-1 MUX cell in synthesis tools.
    * A selector must be a 2-bit unsigned integer. This instrinsic returns v0 when the
    * selector is 0, and returns v3 when the selector is 3.
    * @example {{{
    * v := MuxCell4(sel, v3, v2, v1, v0)
    * }}}
    */

  def apply[T <: Data](sel: UInt, v3: T, v2: T, v1: T, v0: T): T = {
    val inst = Module(new MuxCell4Intrinsic(chisel3.chiselTypeOf(v3)))
    inst.sel := sel
    inst.v3 := v3
    inst.v2 := v2
    inst.v1 := v1
    inst.v0 := v0
    inst.out
  }
}
