// See LICENSE for license details.


package firrtl.constraint

// Is case class because writing tests is easier due to equality is not object equality
case class IsAdd private (known: Option[IsKnown],
                          maxs: Vector[IsMax],
                          mins: Vector[IsMin],
                          others: Vector[Constraint]) extends Constraint with MultiAry {

  def op(b1: IsKnown, b2: IsKnown): IsKnown = b1 + b2

  lazy val children: Vector[Constraint] = {
    if(known.nonEmpty) known.get +: (maxs ++ mins ++ others) else maxs ++ mins ++ others
  }

  def addChild(x: Constraint): IsAdd = x match {
    case k: IsKnown => new IsAdd(merge(Some(k), known), maxs, mins, others)
    case add: IsAdd => new IsAdd(merge(known, add.known), maxs ++ add.maxs, mins ++ add.mins, others ++ add.others)
    case max: IsMax => new IsAdd(known, maxs :+ max, mins, others)
    case min: IsMin => new IsAdd(known, maxs, mins :+ min, others)
    case other      => new IsAdd(known, maxs, mins, others :+ other)
  }

  override def serialize: String = "(" + children.map(_.serialize).mkString(" + ") + ")"

  override def map(f: Constraint=>Constraint): Constraint = IsAdd(children.map(f))

  def reduce(): Constraint = {
    if(children.size == 1) children.head else {
      (known, maxs, mins, others) match {
        case (Some(k), _, _, _) if k.value == 0         => new IsAdd(None, maxs, mins, others).reduce()
        case (Some(k), Vector(max), Vector(), Vector()) => max.map { o => IsAdd(k, o) }.reduce()
        case (Some(k), Vector(), Vector(min), Vector()) => min.map { o => IsAdd(k, o) }.reduce()
        case _ => this
      }
    }
  }
}

object IsAdd {
  def apply(left: Constraint, right: Constraint): Constraint = (left, right) match {
    case (l: IsKnown, r: IsKnown) => l + r
    case _ => apply(Seq(left, right))
  }
  def apply(children: Seq[Constraint]): Constraint = {
    children.foldLeft(new IsAdd(None, Vector(), Vector(), Vector())) { (add, c) =>
      add.addChild(c)
    }.reduce()
  }
}