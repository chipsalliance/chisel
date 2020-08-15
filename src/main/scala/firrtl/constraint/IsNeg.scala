// See LICENSE for license details.

package firrtl.constraint

object IsNeg {
  def apply(child: Constraint): Constraint = new IsNeg(child, 0).reduce()
}

// Dummy arg is to get around weird Scala issue that can't differentiate between a
//   private constructor and public apply that share the same arguments
case class IsNeg private (child: Constraint, dummyArg: Int) extends Constraint {
  override def reduce(): Constraint = child match {
    case k: IsKnown => k.neg
    case x: IsAdd   => IsAdd(x.children.map { b => IsNeg(b) })
    case x: IsMul   => IsMul(Seq(IsNeg(x.children.head)) ++ x.children.tail)
    case x: IsNeg   => x.child
    case x: IsPow   => this
    // -[max(a, b)] -> min[-a, -b]
    case x: IsMax => IsMin(x.children.map { b => IsNeg(b) })
    case x: IsMin => IsMax(x.children.map { b => IsNeg(b) })
    case x: IsVar => this
    case _ => this
  }

  lazy val children = Vector(child)

  override def map(f: Constraint => Constraint): Constraint = IsNeg(f(child))

  override def serialize: String = "(-" + child.serialize + ")"
}
