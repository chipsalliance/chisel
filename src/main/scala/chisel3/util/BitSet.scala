// SPDX-License-Identifier: Apache-2.0

package chisel3.util

/** A Set of [[BitPat]] represents a set of bit vector with mask. */
trait BitSet { outer =>

  /** all [[BitPat]] elements in [[terms]] make up this [[BitSet]].
    * all [[terms]] should be have the same width.
    */
  def terms: Set[BitPat]

  /** Get specified width of said BitSet */
  def getWidth: Int = {
    require(terms.map(_.width).size <= 1, s"All BitPats must be the same size! Got $this")
    // set width = 0 if terms is empty.
    terms.headOption.map(_.width).getOrElse(0)
  }

  override def toString: String = terms.toSeq.sortBy((t: BitPat) => (t.mask, t.value)).mkString("\n")

  /** whether this [[BitSet]] is empty (i.e. no value matches) */
  def isEmpty: Boolean = terms.forall(_.isEmpty)

  /** Check whether this [[BitSet]] overlap with that [[BitSet]], i.e. !(intersect.isEmpty)
    *
    * @param that [[BitSet]] to be checked.
    * @return true if this and that [[BitSet]] have overlap.
    */
  def overlap(that: BitSet): Boolean =
    !terms.flatMap(a => that.terms.map(b => (a, b))).forall { case (a, b) => !a.overlap(b) }

  /** Check whether this [[BitSet]] covers that (i.e. forall b matches that, b also matches this)
    *
    * @param that [[BitSet]] to be covered
    * @return true if this [[BitSet]] can cover that [[BitSet]]
    */
  def cover(that: BitSet): Boolean =
    that.subtract(this).isEmpty

  /** Intersect `this` and `that` [[BitSet]].
    *
    * @param that [[BitSet]] to be intersected.
    * @return a [[BitSet]] containing all elements of `this` that also belong to `that`.
    */
  def intersect(that: BitSet): BitSet =
    terms
      .flatMap(a => that.terms.map(b => a.intersect(b)))
      .filterNot(_.isEmpty)
      .fold(BitSet.empty)(_.union(_))

  /** Subtract that from this [[BitSet]].
    *
    * @param that subtrahend [[BitSet]].
    * @return a [[BitSet]] containing elements of `this` which are not the elements of `that`.
    */
  def subtract(that: BitSet): BitSet =
    terms.map { a =>
      that.terms.map(b => a.subtract(b)).fold(a)(_.intersect(_))
    }.filterNot(_.isEmpty).fold(BitSet.empty)(_.union(_))

  /** Union this and that [[BitSet]]
    *
    * @param that [[BitSet]] to union.
    * @return a [[BitSet]] containing all elements of `this` and `that`.
    */
  def union(that: BitSet): BitSet = new BitSet {
    def terms = outer.terms ++ that.terms
  }

  /** Test whether two [[BitSet]] matches the same set of value
    *
    * @note
    * This method can be very expensive compared to ordinary == operator between two Objects
    *
    * @return true if two [[BitSet]] is same.
    */
  override def equals(obj: Any): Boolean = {
    obj match {
      case that: BitSet => this.getWidth == that.getWidth && this.cover(that) && that.cover(this)
      case _ => false
    }
  }
}

object BitSet {

  /** Construct a [[BitSet]] from a sequence of [[BitPat]].
    * All [[BitPat]] must have the same width.
    */
  def apply(bitpats: BitPat*): BitSet = {
    val bs = new BitSet { def terms = bitpats.flatMap(_.terms).toSet }
    // check width
    bs.getWidth
    bs
  }

  /** Empty [[BitSet]]. */
  val empty: BitSet = new BitSet {
    def terms = Set()
  }

  /** Construct a [[BitSet]] from String.
    * each line should be a valid [[BitPat]] string with the same width.
    */
  def fromString(str: String): BitSet = {
    val bs = new BitSet { def terms = str.split('\n').map(str => BitPat(str)).toSet }
    // check width
    bs.getWidth
    bs
  }
}
