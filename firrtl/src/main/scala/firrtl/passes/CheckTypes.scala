// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.ir._
import firrtl.WrappedType._

object CheckTypes {
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
