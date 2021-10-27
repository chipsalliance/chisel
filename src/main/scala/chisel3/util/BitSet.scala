package chisel3.util

trait BitSetFamily {
  val terms: Set[BitSet]
  assert(terms.map(_.width).size < 1)

  // set width = 0 if terms is empty.
  val width: Int = terms.headOption.map(_.width).getOrElse(0)

  override def toString: String = terms.toSeq.sortBy((t: BitSet) => (t.onSet, t.offSet)).mkString("\n")

  def cover(that: BitSetFamily): Boolean =
    if (terms.isEmpty)
      that.terms.isEmpty
    else
      terms.flatMap(a => that.terms.map(b => (a, b))).foldLeft(true) {
        case (left, (a, b)) => a.cover(b) & left
      }

  override def equals(obj: Any): Boolean = {
    obj match {
      case that: BitSetFamily => if (width == that.width) this.cover(that) && that.cover(this) else false
      case _ => false
    }
  }
}

trait BitSet extends BitSetFamily {
  override val terms = Set(this)
  val onSet:  BigInt
  val offSet: BigInt
  val width:  Int

  assert(width > 0)

  def cover(that: BitSet) = ((onSet & ~that.onSet) == 0) & ((offSet & ~that.offSet) == 0)

  override def toString: String = Seq
    .tabulate(width) { i =>
      (onSet.testBit(i), offSet.testBit(i)) match {
        case (true, true)   => "-"
        case (true, false)  => "1"
        case (false, true)  => "0"
        case (false, false) => "~"
      }
    }
    .mkString
}
