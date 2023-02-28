// SPDX-License-Identifier: Apache-2.0

package firrtl.constraint

import firrtl.ir.Closed

object IsMul {
  def apply(left: Constraint, right: Constraint): Constraint = (left, right) match {
    case (l: IsKnown, r: IsKnown) => l * r
    case _ => apply(Seq(left, right))
  }
  def apply(children: Seq[Constraint]): Constraint = {
    children
      .foldLeft(new IsMul(None, Vector())) { (add, c) =>
        add.addChild(c)
      }
      .reduce()
  }
}

case class IsMul private (known: Option[IsKnown], others: Vector[Constraint]) extends MultiAry {

  def op(b1: IsKnown, b2: IsKnown): IsKnown = b1 * b2

  lazy val children: Vector[Constraint] = if (known.nonEmpty) known.get +: others else others

  def addChild(x: Constraint): IsMul = x match {
    case k:   IsKnown => new IsMul(known = merge(Some(k), known), others)
    case mul: IsMul   => new IsMul(merge(known, mul.known), others ++ mul.others)
    case other => new IsMul(known, others :+ other)
  }

  override def reduce(): Constraint = {
    if (children.size == 1) children.head
    else {
      (known, others) match {
        case (Some(Closed(x)), _) if x == BigDecimal(1) => new IsMul(None, others).reduce()
        case (Some(Closed(x)), _) if x == BigDecimal(0) => Closed(0)
        case (Some(Closed(x)), Vector(m: IsMax)) if x > 0 =>
          IsMax(m.children.map { c => IsMul(Closed(x), c) })
        case (Some(Closed(x)), Vector(m: IsMax)) if x < 0 =>
          IsMin(m.children.map { c => IsMul(Closed(x), c) })
        case (Some(Closed(x)), Vector(m: IsMin)) if x > 0 =>
          IsMin(m.children.map { c => IsMul(Closed(x), c) })
        case (Some(Closed(x)), Vector(m: IsMin)) if x < 0 =>
          IsMax(m.children.map { c => IsMul(Closed(x), c) })
        case _ => this
      }
    }
  }

  override def map(f: Constraint => Constraint): Constraint = IsMul(children.map(f))

  override def serialize: String = "(" + children.map(_.serialize).mkString(" * ") + ")"
}
