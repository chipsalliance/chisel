// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._
import Utils.trim
import firrtl.constraint.Constraint

object Implicits {
  implicit def constraint2bound(c: Constraint): Bound = c match {
    case x: Bound => x
    case x => CalcBound(x)
  }
  implicit def constraint2width(c: Constraint): Width = c match {
    case Closed(x) if trim(x).isWhole => IntWidth(x.toBigInt)
    case x                            => CalcWidth(x)
  }
  implicit def width2constraint(w: Width): Constraint = w match {
    case CalcWidth(x: Constraint) => x
    case IntWidth(x)  => Closed(BigDecimal(x))
    case UnknownWidth => UnknownBound
    case v: Constraint => v
  }
}
