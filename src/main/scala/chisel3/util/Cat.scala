// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.core.SeqUtils

object Cat {
  /** Concatenates the argument data elements, in argument order, together.
    */
  def apply[T <: Bits](a: T, r: T*): UInt = apply(a :: r.toList)

  /** Concatenates the data elements of the input sequence, in reverse sequence order, together.
    * The first element of the sequence forms the most significant bits, while the last element
    * in the sequence forms the least significant bits.
    *
    * Equivalent to r(0) ## r(1) ## ... ## r(n-1).
    */
  def apply[T <: Bits](r: Seq[T]): UInt = SeqUtils.asUInt(r.reverse)
}
