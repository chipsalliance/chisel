// SPDX-License-Identifier: Apache-2.0

package firrtl.constraint

object IsFloor {
  def apply(child: Constraint): Constraint = new IsFloor(child, 0).reduce()
}

case class IsFloor private (child: Constraint, dummyArg: Int) extends Constraint {

  override def reduce(): Constraint = child match {
    case k: IsKnown => k.floor
    case x: IsAdd   => this
    case x: IsMul   => this
    case x: IsNeg   => this
    case x: IsPow   => this
    // floor(max(a, b)) -> max(floor(a), floor(b))
    case x: IsMax => IsMax(x.children.map { b => IsFloor(b) })
    case x: IsMin => IsMin(x.children.map { b => IsFloor(b) })
    case x: IsVar => this
    // floor(floor(x)) -> floor(x)
    case x: IsFloor => x
    case _ => this
  }
  val children = Vector(child)

  override def map(f: Constraint => Constraint): Constraint = IsFloor(f(child))

  override def serialize: String = "floor(" + child.serialize + ")"
}
