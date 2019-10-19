// See LICENSE for license details.

package firrtl

import firrtl.ir._
import Utils.trim
import firrtl.constraint.Constraint

object Implicits {
  implicit def int2WInt(i: Int): WrappedInt = WrappedInt(BigInt(i))
  implicit def bigint2WInt(i: BigInt): WrappedInt = WrappedInt(i)
  implicit def constraint2bound(c: Constraint): Bound = c match {
    case x: Bound => x
    case x => CalcBound(x)
  }
  implicit def constraint2width(c: Constraint): Width = c match {
    case Closed(x) if trim(x).isWhole => IntWidth(x.toBigInt)
    case x => CalcWidth(x)
  }
  implicit def width2constraint(w: Width): Constraint = w match {
    case CalcWidth(x: Constraint) => x
    case IntWidth(x) => Closed(BigDecimal(x))
    case UnknownWidth => UnknownBound
    case v: Constraint => v
  }
}
case class WrappedInt(value: BigInt) {
  def U: Expression = UIntLiteral(value, IntWidth(Utils.getUIntWidth(value)))
  def S: Expression = SIntLiteral(value, IntWidth(Utils.getSIntWidth(value)))
}
