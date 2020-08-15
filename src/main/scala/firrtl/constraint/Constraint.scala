// See LICENSE for license details.

package firrtl.constraint

/** Trait for all Constraint Solver expressions */
trait Constraint {
  def serialize: String
  def map(f: Constraint => Constraint): Constraint
  val children: Vector[Constraint]
  def reduce(): Constraint
}

/** Trait for constraints with more than one argument */
trait MultiAry extends Constraint {
  def op(a:     IsKnown, b:          IsKnown): IsKnown
  def merge(b1: Option[IsKnown], b2: Option[IsKnown]): Option[IsKnown] = (b1, b2) match {
    case (Some(x), Some(y)) => Some(op(x, y))
    case (_, y: Some[_]) => y
    case (x: Some[_], _) => x
    case _ => None
  }
}
