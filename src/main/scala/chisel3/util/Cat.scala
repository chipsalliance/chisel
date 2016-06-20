// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.core.SeqUtils

object Cat {
  /** Combine data elements together
    * @param a Data to combine with
    * @param r any number of other Data elements to be combined in order
    * @return A UInt which is all of the bits combined together
    */
  def apply[T <: Bits](a: T, r: T*): UInt = apply(a :: r.toList)

  /** Combine data elements together
    * @param r any number of other Data elements to be combined in order
    * @return A UInt which is all of the bits combined together
    */
  def apply[T <: Bits](r: Seq[T]): UInt = SeqUtils.asUInt(r.reverse)
}
