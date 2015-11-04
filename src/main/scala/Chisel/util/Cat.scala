// See LICENSE for license details.

package Chisel

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
      left ## right
    }
  }
}
