// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._

object BinaryToGray {

  /** Turns a binary number into gray code. */
  def apply(in: UInt): UInt = in ^ (in >> 1)
}

object GrayToBinary {

  /** Inverts the [[BinaryToGray]] operation. */
  def apply(in: UInt, width: Int): UInt = apply(in(width - 1, 0))

  /** Inverts the [[BinaryToGray]] operation. */
  def apply(in: UInt): UInt = if (in.getWidth < 2) { in }
  else {
    val bits = in.getWidth - 2 to 0 by -1
    Cat(bits.scanLeft(in.head(1)) { case (prev, ii) => prev ^ in(ii) })
  }
}
