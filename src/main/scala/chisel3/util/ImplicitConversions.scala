// See LICENSE for license details.

package chisel3.util

import chisel3._

object ImplicitConversions {
  implicit def intToUInt(x: Int): UInt = UInt.Lit(x)
  implicit def booleanToBool(x: Boolean): Bool = Bool(x)
}
