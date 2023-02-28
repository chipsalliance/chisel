// SPDX-License-Identifier: Apache-2.0

package firrtl.constraint

object IsMin {
  def apply(left: Constraint, right: Constraint): Constraint = (left, right) match {
    case (l: IsKnown, r: IsKnown) => l.min(r)
    case _ => apply(Seq(left, right))
  }
  def apply(children: Seq[Constraint]): Constraint = {
    children
      .foldLeft(new IsMin(None, Vector(), Vector())) { (add, c) =>
        add.addChild(c)
      }
      .reduce()
  }
}

case class IsMin private[constraint] (known: Option[IsKnown], maxs: Vector[IsMax], others: Vector[Constraint])
    extends MultiAry {

  def op(b1: IsKnown, b2: IsKnown): IsKnown = b1.min(b2)

  override def serialize: String = "min(" + children.map(_.serialize).mkString(", ") + ")"

  override def map(f: Constraint => Constraint): Constraint = IsMin(children.map(f))

  lazy val children: Vector[Constraint] = {
    if (known.nonEmpty) known.get +: (maxs ++ others) else maxs ++ others
  }

  def reduce(): Constraint = {
    if (children.size == 1) children.head
    else {
      (known, maxs, others) match {
        case (Some(IsKnown(i)), _, _) =>
          // Eliminate maximums who have a known maximum value which is larger than known minimum value
          val filteredMaxs = maxs.filter {
            case IsMax(Some(IsKnown(a)), _, _) if a >= i => false
            case other                                   => true
          }
          // If a successful filter, rerun reduce
          val newMin = new IsMin(known, filteredMaxs, others)
          if (filteredMaxs.size != maxs.size) {
            newMin.reduce()
          } else newMin
        case _ => this
      }
    }
  }

  def addChild(x: Constraint): IsMin = x match {
    case k:   IsKnown => new IsMin(merge(Some(k), known), maxs, others)
    case max: IsMax   => new IsMin(known, maxs :+ max, others)
    case min: IsMin   => new IsMin(merge(min.known, known), maxs ++ min.maxs, others ++ min.others)
    case other => new IsMin(known, maxs, others :+ other)
  }
}
