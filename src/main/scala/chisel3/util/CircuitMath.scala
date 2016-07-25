// See LICENSE for license details.

/** Circuit-land math operations.
  */

package chisel3.util

import chisel3._

/** Compute the base-2 integer logarithm of a UInt
  * @example
  * {{{ data_out := Log2(data_in) }}}
  * @note The result is truncated, so e.g. Log2(13.U) = 3
  */
object Log2 {
  /** Compute the Log2 on the least significant n bits of x */
  def apply(x: Bits, width: Int): UInt = {
    if (width < 2) {
      UInt(0)
    } else if (width == 2) {
      x(1)
    } else if (width <= divideAndConquerThreshold) {
      Mux(x(width-1), UInt(width-1), apply(x, width-1))
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
