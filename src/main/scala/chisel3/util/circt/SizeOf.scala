// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.SourceInfo

object SizeOf {

  /** Creates an intrinsic which returns the size of a type.  The returned size
    * is after width inference, so you can use this to compute expressions based
    * on the inferred size of types.
    *
    * @example {{{
    * val a = Wire(UInt())
    * a := 1 << (SizeOf(a) - 1)
    * }}}
    */
  def apply[T <: Data](gen: T)(implicit sourceInfo: SourceInfo): Data = {
    IntrinsicExpr("circt_sizeof", UInt(32.W))(gen)
  }
}
