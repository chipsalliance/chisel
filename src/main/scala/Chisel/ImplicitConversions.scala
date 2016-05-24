// See LICENSE for license details.

package Chisel

object ImplicitConversions {
  implicit def intToUInt(x: Int): UInt = UInt.Lit(x)
  implicit def booleanToBool(x: Boolean): Bool = Bool.Lit(x)
}
