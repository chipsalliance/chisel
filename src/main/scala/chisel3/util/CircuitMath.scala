// SPDX-License-Identifier: Apache-2.0

/** Circuit-land math operations.
  */

package chisel3.util

import chisel3._

/** Returns the base-2 integer logarithm of an UInt.
  *
  * @note The result is truncated, so e.g. Log2(13.U) === 3.U
  *
  * @example {{{
  * Log2(8.U)  // evaluates to 3.U
  * Log2(13.U)  // evaluates to 3.U (truncation)
  * Log2(myUIntWire)
  * }}}
  */
object Log2 {

  /** Returns the base-2 integer logarithm of the least-significant `width` bits of an UInt.
    */
  def apply(x: Bits, width: Int): UInt = {
    if (width < 2) {
      0.U
    } else if (width == 2) {
      x(1)
    } else if (width <= divideAndConquerThreshold) {
      Mux(x(width - 1), (width - 1).asUInt, apply(x, width - 1))
    } else {
      val mid = 1 << (log2Ceil(width) - 1)
      val hi = x(width - 1, mid)
      val lo = x(mid - 1, 0)
      val useHi = hi.orR
      Cat(useHi, Mux(useHi, Log2(hi, width - mid), Log2(lo, mid)))
    }
  }

  def apply(x: Bits): UInt = apply(x, x.getWidth)

  private def divideAndConquerThreshold = 4
}

/** Create a carry save adder built from full adders. If more then 3 input terms, construct a Wallace tree.
 *
 *  The function returns 2 output terms, which are still to be added by a carry-propagate adder.
 *  Future improvement idea: add support for negative weights at arbitrary bit positions; required to support subtraction.
 *  Function can return two UInt hardware terms together with a signed constant offset, computed at elaboration-time.
 */

object Csa {

  /** Create a carry save adder built from full adders. If more then 3 input terms, construct a Wallace tree.
   *  The function returns 2 output terms (unless called with less than 2 inputs, then it returns 1).
   *  The outputs are still to be added by a carry-propagate adder.
   */
  def apply(x: Seq[UInt]): Seq[UInt] = {
    x.length match {
      case 0 => Seq(0.U(0.W))
      case 1 => x
      case 2 => x
      case 3 => Seq(x(0) ^ x(1) ^ x(2), (x(0) & x(1) | x(0) & x(2) | x(1) & x(2)) << 1) // sum, carry
      case _ =>
        Csa(x.grouped(3).map(xyz => Csa(xyz)).reduce(_ ++ _)) // every group of 3 reduces to 2. Result to next level
    }
  }
}
