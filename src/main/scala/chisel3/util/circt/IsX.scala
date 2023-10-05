// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.IntrinsicModule
import chisel3.internal.Builder

import circt.Intrinsic

/** Create a module with a parameterized type which returns whether the input
  * is a verilog 'x'.
  */
private class IsXIntrinsic[T <: Data](gen: T) extends IntrinsicModule("circt_isX") {
  val i = IO(Input(gen))
  val found = IO(Output(Bool()))
}

object IsX {

  /** Creates an intrinsic which returns whether the input is a verilog 'x'.
    *
    * @example {{{
    * b := IsX(a)
    * }}}
    */
  def apply[T <: Data](gen: T): Bool = {
    val inst = Module(new IsXIntrinsic(chiselTypeOf(gen)))
    inst.i := gen
    inst.found
  }
}
