// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3._

object util {
  private[chisel3] def _padHandleBool[A <: Bits](
    x:     A,
    width: Int
  ): A = ???

  private[chisel3] def _resizeToWidth[A <: Bits](
    that:           A,
    targetWidthOpt: Option[Int]
  )(
    fromUInt:       UInt => A
  ): A = ???
}
