// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.Builder

object IsX {

  /** Creates an intrinsic which returns whether the input is a verilog 'x'.
    *
    * @example {{{
    * b := IsX(a)
    * }}}
    */
  def apply[T <: Data](gen: T)(implicit sourceInfo: SourceInfo): Bool = {
    IntrinsicExpr("circt_isX", Bool())(gen)
  }
}
