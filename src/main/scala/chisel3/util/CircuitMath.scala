// See LICENSE for license details.

/** Circuit-land math operations.
  */

package chisel3.util

import chisel3._

/** Compute the base-2 integer logarithm of a UInt
  * @example
  * {{{ data_out := Log2(data_in) }}}
  * @note The result is truncated, so e.g. Log2(UInt(13)) = 3
  */
object Log2 {
  /** Compute the Log2 on the least significant n bits of x */
  def apply(x: Bits, width: Int): UInt = {
    if (width < 2) {
      UInt(0)
    } else if (width == 2) {
      x(1)
    } else {
      Mux(x(width-1), UInt(width-1), apply(x, width-1))
    }
  }

  def apply(x: Bits): UInt = apply(x, x.getWidth)
}
