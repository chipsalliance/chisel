// SPDX-License-Identifier: Apache-2.0

package chisel3.util

trait BitSetFamily { bsf =>
  val terms: Set[BitSet]
  assert(terms.map(_.width).size < 1)

  // set width = 0 if terms is empty.
  val width: Int = terms.headOption.map(_.width).getOrElse(0)

  override def toString: String = terms.toSeq.sortBy((t: BitSet) => (t.onSet, t.offSet)).mkString("\n")

  def isEmpty: Boolean = terms.forall(_.isEmpty)

  def cover(that: BitSetFamily): Boolean =
    if (terms.isEmpty)
      that.terms.isEmpty
    else
      terms.flatMap(a => that.terms.map(b => (a, b))).foldLeft(true) {
        case (left, (a, b)) => a.cover(b) & left
      }

  def intersect(that: BitSetFamily): BitSetFamily = new BitSetFamily {
    val terms = bsf.terms.flatMap(a => that.terms.map(b => a.intersect(b))).filterNot(_.isEmpty)
  }

  def subtract(that: BitSetFamily): BitSetFamily = new BitSetFamily {
    val terms = bsf.terms.flatMap(a => that.terms.map(b => a.subtract(b))).filterNot(_.isEmpty)
  }

  def union(that: BitSetFamily): BitSetFamily = new BitSetFamily {
    val terms = bsf.terms ++ that.terms
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case that: BitSetFamily => if (width == that.width) this.cover(that) && that.cover(this) else false
      case _ => false
    }
  }
}

trait BitSet extends BitSetFamily { b =>
  override val terms = Set(this)
  val onSet:  BigInt
  val offSet: BigInt
  val width:  Int

  assert(width > 0)

  def cover(that: BitSet) = ((onSet & that.onSet) == that.onSet) & ((offSet & that.offSet) == that.offSet)

  def intersect(that: BitSet): BitSet = {
    require(width == that.width)
    new BitSet {
      val onSet:          BigInt = b.onSet & that.onSet
      val offSet:         BigInt = b.offSet & that.offSet
      override val width: Int = b.width
    }
  }

  def subtract(that: BitSet): BitSet = {
    require(width == that.width)
    new BitSet {
      val onSet:          BigInt = b.onSet & ~that.onSet
      val offSet:         BigInt = b.offSet & ~that.offSet
      override val width: Int = b.width
    }
  }

  override def isEmpty: Boolean = (onSet | offSet) == 0
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
