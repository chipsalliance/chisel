// See LICENSE for license details.

package firrtl.constraint

/** Represents either greater or equal to or less than or equal to
  * Is passed to the constraint solver to resolve
  */
trait Inequality {
  def left:  String
  def right: Constraint
  def geq:   Boolean
}

case class GreaterOrEqual(left: String, right: Constraint) extends Inequality {
  val geq = true
  override def toString: String = s"$left >= ${right.serialize}"
}

case class LesserOrEqual(left: String, right: Constraint) extends Inequality {
  val geq = false
  override def toString: String = s"$left <= ${right.serialize}"
}
