// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._

/** Map each bit to the logical OR of itself and all bits with lower index
  *
  * Here `scanLeft` means "start from the left and iterate to the right, where left is the lowest index", a common operation on arrays and lists.
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

/** Map each bit to the logical OR of itself and all bits with higher index
  *
  * Here `scanRight` means "start from the right and iterate to the left, where right is the highest index", a common operation on arrays and lists.
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
