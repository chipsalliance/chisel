// See LICENSE for license details.

package Chisel
import Builder.pushOp
import PrimOp._

// REVIEW TODO: Should the FIRRTL emission be part of Bits, with a separate
// Cat in stdlib that can do a reduction among multiple elements?
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
  def apply[T <: Bits](r: Seq[T]): UInt = {
    if (r.tail.isEmpty) {
      r.head.asUInt
    } else {
      val left = apply(r.slice(0, r.length/2))
      val right = apply(r.slice(r.length/2, r.length))
      val w = left.width + right.width
      pushOp(DefPrim(UInt(w), ConcatOp, left.ref, right.ref))
    }
  }
}
