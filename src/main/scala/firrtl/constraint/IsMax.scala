// SPDX-License-Identifier: Apache-2.0

package firrtl.constraint

object IsMax {
  def apply(left: Constraint, right: Constraint): Constraint = (left, right) match {
    case (l: IsKnown, r: IsKnown) => l.max(r)
    case _ => apply(Seq(left, right))
  }
  def apply(children: Seq[Constraint]): Constraint = {
    val x = children.foldLeft(new IsMax(None, Vector(), Vector())) { (add, c) =>
      add.addChild(c)
    }
    x.reduce()
  }
}

case class IsMax private[constraint] (known: Option[IsKnown], mins: Vector[IsMin], others: Vector[Constraint])
    extends MultiAry {

  def op(b1: IsKnown, b2: IsKnown): IsKnown = b1.max(b2)

  override def serialize: String = "max(" + children.map(_.serialize).mkString(", ") + ")"

  override def map(f: Constraint => Constraint): Constraint = IsMax(children.map(f))

  lazy val children: Vector[Constraint] = {
    if (known.nonEmpty) known.get +: (mins ++ others) else mins ++ others
  }

  def reduce(): Constraint = {
    if (children.size == 1) children.head
    else {
      (known, mins, others) match {
        case (Some(IsKnown(a)), _, _) =>
          // Eliminate minimums who have a known minimum value which is smaller than known maximum value
          val filteredMins = mins.filter {
            case IsMin(Some(IsKnown(i)), _, _) if i <= a => false
            case other                                   => true
          }
          // If a successful filter, rerun reduce
          val newMax = new IsMax(known, filteredMins, others)
          if (filteredMins.size != mins.size) {
            newMax.reduce()
          } else newMax
        case _ => this
      }
    }
  }

  def addChild(x: Constraint): IsMax = x match {
    case k:   IsKnown => new IsMax(known = merge(Some(k), known), mins, others)
    case max: IsMax   => new IsMax(known = merge(known, max.known), max.mins ++ mins, others ++ max.others)
    case min: IsMin   => new IsMax(known, mins :+ min, others)
    case other => new IsMax(known, mins, others :+ other)
  }
}
