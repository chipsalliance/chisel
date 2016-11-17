// See LICENSE for license details.

package chisel3.util

import chisel3._

object ImplicitConversions {
  implicit def intToUInt(x: Int): UInt = chisel3.core.fromIntToLiteral(x).asUInt
  implicit def booleanToBool(x: Boolean): Bool = x.asBool
}
