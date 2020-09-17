// SPDX-License-Identifier: Apache-2.0

package firrtl.constraint

object IsPow {
  def apply(child: Constraint): Constraint = new IsPow(child, 0).reduce()
}

// Dummy arg is to get around weird Scala issue that can't differentiate between a
//   private constructor and public apply that share the same arguments
case class IsPow private (child: Constraint, dummyArg: Int) extends Constraint {
  override def reduce(): Constraint = child match {
    case k: IsKnown => k.pow
    // 2^(a + b) -> 2^a * 2^b
    case x: IsAdd => IsMul(x.children.map { b => IsPow(b) })
    case x: IsMul => this
    case x: IsNeg => this
    case x: IsPow => this
    // 2^(max(a, b)) -> max(2^a, 2^b) since two is always positive, so a, b control magnitude
    case x: IsMax => IsMax(x.children.map { b => IsPow(b) })
    case x: IsMin => IsMin(x.children.map { b => IsPow(b) })
    case x: IsVar => this
    case _ => this
  }

  val children = Vector(child)

  override def map(f: Constraint => Constraint): Constraint = IsPow(f(child))

  override def serialize: String = "(2^" + child.serialize + ")"
}
