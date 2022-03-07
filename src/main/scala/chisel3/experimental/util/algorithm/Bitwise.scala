// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.util.algorithm

import chisel3._

/** Map each bits to logical or of itself and all bits with lower index.
  * Here `scanLeft` means "start at the left and look to the right, where left is the lowest index", a common operation on arrays and lists.
  * This is consistent with the `left` as in "shift left" performed on bits, which means "start at the right (least significant bit) and move to the left".
  * @example {{{
  * scanLeftOr("b00001000".U(8.W)) // Returns "b11111000".U
  * scanLeftOr("b00010100".U(8.W)) // Returns "b11111100".U
  * scanLeftOr("b00000000".U(8.W)) // Returns "b00000000".U
  * }}}
  */
object scanLeftOr {
  def apply(data: UInt): UInt = {
    val width = data.widthOption match {
      case Some(w) => w
      case None    => throw new IllegalArgumentException("Cannot call scanLeftOr on data with unknown width.")
    }

    def helper(s: Int, x: UInt): UInt =
      if (s >= width) x else helper(s + s, x | (x << s)(width - 1, 0))
    helper(1, data)(width - 1, 0)
  }
}
/** Map each bits to logical or of itself and all bits with higher index. 
  * Here `scanRight` means "start at the right and look to the left, where right is the highest index", a common operation on arrays and lists.
  * This is consistent with the `right` as in "shift right" performed on bits, which means "start at the left (most significant bit) and move to the right".
  * @example {{{
  * scanRightOr("b00001000".U) // Returns "b00001111".U
  * scanRightOr("b00010100".U) // Returns "b00011111".U
  * scanRightOr("b00000000".U) // Returns "b00000000".U
  * }}}
  */
object scanRightOr {
  def apply(data: UInt): UInt = {
    val width = data.widthOption match {
      case Some(w) => w
      case None    => throw new IllegalArgumentException("Cannot call scanRightOr on data with unknown width.")
    }
    def helper(s: Int, x: UInt): UInt =
      if (s >= width) x else helper(s + s, x | (x >> s))
    helper(1, data)(width - 1, 0)
  }
}
