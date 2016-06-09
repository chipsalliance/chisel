// See LICENSE for license details.

package chisel.util

import chisel._

object ImplicitConversions {
  implicit def intToUInt(x: Int): UInt = UInt(x)
  implicit def booleanToBool(x: Boolean): Bool = Bool(x)
}
