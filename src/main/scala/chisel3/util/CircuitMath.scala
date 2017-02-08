// See LICENSE for license details.

/** Circuit-land math operations.
  */

package chisel3.util

import chisel3._
import chisel3.internal.naming.chiselName  // can't use chisel3_ version because of compile order

/** Returns the base-2 integer logarithm of an UInt.
  *
  * @note The result is truncated, so e.g. Log2(13.U) === 3.U
  *
  * @example {{{
  * Log2(8.U)  // evaluates to 3.U
  * Log2(13.U)  // evaluates to 3.U (truncation)
  * Log2(myUIntWire)
  * }}}
  *
  */
object Log2 {
  /** Returns the base-2 integer logarithm of the least-significant `width` bits of an UInt.
    */
  @chiselName
  def apply(x: Bits, width: Int): UInt = {
    if (width < 2) {
      0.U
    } else if (width == 2) {
      x(1)
    } else if (width <= divideAndConquerThreshold) {
      Mux(x(width-1), (width-1).asUInt, apply(x, width-1))
    } else {
      val mid = 1 << (log2Ceil(width) - 1)
      val hi = x(width-1, mid)
      val lo = x(mid-1, 0)
      val useHi = hi.orR
      Cat(useHi, Mux(useHi, Log2(hi, width - mid), Log2(lo, mid)))
    }
  }

  def apply(x: Bits): UInt = apply(x, x.getWidth)

  private def divideAndConquerThreshold = 4
}
