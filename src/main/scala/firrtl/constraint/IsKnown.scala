// See LICENSE for license details.

package firrtl.constraint

object IsKnown {
  def unapply(b: Constraint): Option[BigDecimal] = b match {
    case k: IsKnown => Some(k.value)
    case _ => None
  }
}

/** Constant values must extend this trait see [[firrtl.ir.Closed and firrtl.ir.Open]] */
trait IsKnown extends Constraint {
  val value: BigDecimal

  /** Addition */
  def +(that: IsKnown): IsKnown

  /** Multiplication */
  def *(that: IsKnown): IsKnown

  /** Max */
  def max(that: IsKnown): IsKnown

  /** Min */
  def min(that: IsKnown): IsKnown

  /** Negate */
  def neg: IsKnown

  /** 2 << value */
  def pow: IsKnown

  /** Floor */
  def floor: IsKnown

  override def map(f: Constraint => Constraint): Constraint = this

  val children: Vector[Constraint] = Vector.empty[Constraint]

  def reduce(): IsKnown = this
}
