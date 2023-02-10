// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.ir._
import firrtl.WrappedType._

object CheckTypes {
  // Custom Exceptions
  class InvalidConnect(info: Info, mname: String, con: String, lhs: Expression, rhs: Expression)
      extends PassException({
        val ltpe = s"  ${lhs.serialize}: ${lhs.tpe.serialize}"
        val rtpe = s"  ${rhs.serialize}: ${rhs.tpe.serialize}"
        s"$info: [module $mname]  Type mismatch in '$con'.\n$ltpe\n$rtpe"
      })
  class IllegalResetType(info: Info, mname: String, exp: String)
      extends PassException(
        s"$info: [module $mname]  Register resets must have type Reset, AsyncReset, or UInt<1>: $exp."
      )

  def legalResetType(tpe: Type): Boolean = tpe match {
    case UIntType(IntWidth(w)) if w == 1 => true
    case AsyncResetType                  => true
    case ResetType                       => true
    case UIntType(UnknownWidth)          =>
      // cannot catch here, though width may ultimately be wrong
      true
    case _ => false
  }

  def validConnect(locTpe: Type, expTpe: Type): Boolean = {
    wt(locTpe).superTypeOf(wt(expTpe))
  }
}
