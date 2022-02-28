// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._

import scala.language.implicitConversions

/** Implicit conversions to automatically convert [[scala.Boolean]] and [[scala.Int]] to [[Bool]]
  *  and [[UInt]] respectively
  */
object ImplicitConversions {
  // The explicit fromIntToLiteral resolves an ambiguous conversion between fromIntToLiteral and
  // UInt.asUInt.
  implicit def intToUInt(x:     Int):     UInt = chisel3.fromIntToLiteral(x).asUInt
  implicit def booleanToBool(x: Boolean): Bool = x.B
}
