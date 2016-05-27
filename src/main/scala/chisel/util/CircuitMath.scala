// See LICENSE for license details.

/** Circuit-land math operations.
  */

package chisel.util

import chisel._

/** Compute Log2 with truncation of a UInt in hardware using a Mux Tree
  * An alternative interpretation is it computes the minimum number of bits needed to represent x
  * @example
  * {{{ data_out := Log2(data_in) }}}
  * @note Truncation is used so Log2(UInt(12412)) = 13*/
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
