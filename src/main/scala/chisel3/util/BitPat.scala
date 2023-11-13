// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import scala.language.experimental.macros
import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform
import scala.collection.mutable
import scala.util.hashing.MurmurHash3

object BitPat {

  private[chisel3] implicit val bitPatOrder: Ordering[BitPat] = new Ordering[BitPat] {
    import scala.math.Ordered.orderingToOrdered
    def compare(x: BitPat, y: BitPat): Int = (x.getWidth, x.value, x.mask).compare(y.getWidth, y.value, y.mask)
  }

  /** Parses a bit pattern string into (bits, mask, width).
    *
    * @return bits the literal value, with don't cares being 0
    * @return mask the mask bits, with don't cares being 0 and cares being 1
    * @return width the number of bits in the literal, including values and
    * don't cares, but not including the white space and underscores
    */
  private def parse(x: String): (BigInt, BigInt, Int) = {
    // Notes:
    // While Verilog Xs also handle octal and hex cases, there isn't a
    // compelling argument and no one has asked for it.
    // If ? parsing is to be exposed, the return API needs further scrutiny
    // (especially with things like mask polarity).
    require(x.head == 'b', "BitPats must be in binary and be prefixed with 'b'")
    require(x.length > 1, "BitPat width cannot be 0.")
    var bits = BigInt(0)
    var mask = BigInt(0)
    var count = 0
    for (d <- x.tail) {
      if (!(d == '_' || d.isWhitespace)) {
        require("01?".contains(d), "Literal: " + x + " contains illegal character: " + d)
        mask = (mask << 1) + (if (d == '?') 0 else 1)
        bits = (bits << 1) + (if (d == '1') 1 else 0)
        count += 1
      }
    }

    (bits, mask, count)
  }

  /** Creates a `BitPat` literal from a string.
    *
    * @param n the literal value as a string, in binary, prefixed with 'b'
    * @note legal characters are '0', '1', and '?', as well as '_' and white
    * space (which are ignored)
    */
  def apply(n: String): BitPat = {
    val (bits, mask, width) = parse(n)
    new BitPat(bits, mask, width)
  }

  /** Creates a `BitPat` of all don't cares of the specified bitwidth.
    *
    * @example {{{
    * val myDontCare = BitPat.dontCare(4)  // equivalent to BitPat("b????")
    * }}}
    */
  def dontCare(width: Int): BitPat = BitPat("b" + ("?" * width))

  /** Creates a `BitPat` of all 1 of the specified bitwidth.
    *
    * @example {{{
    * val myY = BitPat.Y(4)  // equivalent to BitPat("b1111")
    * }}}
    */
  def Y(width: Int = 1): BitPat = BitPat("b" + ("1" * width))

  /** Creates a `BitPat` of all 0 of the specified bitwidth.
    *
    * @example {{{
    * val myN = BitPat.N(4)  // equivalent to BitPat("b0000")
    * }}}
    */
  def N(width: Int = 1): BitPat = BitPat("b" + ("0" * width))

  /** Allows BitPats to be used where a UInt is expected.
    *
    * @note the BitPat must not have don't care bits (will error out otherwise)
    */
  def bitPatToUInt(x: BitPat): UInt = {
    require(x.mask == (BigInt(1) << x.getWidth) - 1)
    x.value.asUInt(x.getWidth.W)
  }

  /** Allows UInts to be used where a BitPat is expected, useful for when an
    * interface is defined with BitPats but not all cases need the partial
    * matching capability.
    *
    * @note the UInt must be a literal
    */
  def apply(x: UInt): BitPat = {
    require(x.isLit, s"$x is not a literal, BitPat.apply(x: UInt) only accepts literals")
    val len = if (x.isWidthKnown) x.getWidth else 0
    apply("b" + x.litValue.toString(2).reverse.padTo(len, "0").reverse.mkString)
  }

  implicit class fromUIntToBitPatComparable(x: UInt) extends SourceInfoDoc {

    import scala.language.experimental.macros

    final def ===(that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def =/=(that: BitPat): Bool = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_===(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that === x

    /** @group SourceInfoTransformMacro */
    def do_=/=(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that =/= x
  }
}

package experimental {
  object BitSet {

    /** Construct a `BitSet` from a sequence of BitPat`.
      * All `BitPat` must have the same width.
      */
    def apply(bitpats: BitPat*): BitSet = {
      val bs = new BitSet { def terms = bitpats.flatMap(_.terms).toSet }
      // check width
      bs.getWidth
      bs
    }

    /** Empty `BitSet`. */
    val empty: BitSet = new BitSet {
      def terms = Set()
    }

    /** Construct a `BitSet` from String.
      * each line should be a valid `BitPat` string with the same width.
      */
    def fromString(str: String): BitSet = {
      val bs = new BitSet { def terms = str.split('\n').map(str => BitPat(str)).toSet }
      // check width
      bs.getWidth
      bs
    }

    /** Construct a `BitSet` matching a range of value
      * automatically infer width by the bit length of (start + length - 1)
      *
      * @param start The smallest matching value
      * @param length The length of the matching range
      * @return A `BitSet` matching exactly all inputs in range [start, start + length)
      */
    def fromRange(
      start:  BigInt,
      length: BigInt
    ): BitSet = fromRange(start, length, (start + length - 1).bitLength)

    /** Construct a `BitSet` matching a range of value
      *
      * @param start The smallest matching value
      * @param length The length of the matching range
      * @param width The width of the constructed `BitSet`. If not given, the returned `BitSet` have the width of the maximum possible matching value.
      * @return A `BitSet` matcing exactly all inputs in range [start, start + length)
      */
    def fromRange(
      start:  BigInt,
      length: BigInt,
      width:  Int
    ): BitSet = {
      require(length > 0, "Cannot construct a empty BitSetRange")
      val maxKnownLength = (start + length - 1).bitLength
      require(
        width >= maxKnownLength,
        s"Cannot construct a BitSetRange with width($width) smaller than its range end(b${(start + length - 1).toString(2)})"
      )

      // Break down to individual bitpats
      val atoms = {
        val collected = mutable.Set[BitPat]()
        var ptr = start
        var left = length
        while (left > 0) {
          var curPow = left.bitLength - 1
          if (ptr != 0) {
            val maxPow = ptr.lowestSetBit
            if (maxPow < curPow) curPow = maxPow
          }

          val inc = BigInt(1) << curPow
          require((ptr & inc - 1) == 0, "BitPatRange: Internal sanity check")
          val mask = (BigInt(1) << width) - inc
          collected.add(new BitPat(ptr, mask, width))
          ptr += inc
          left -= inc
        }

        collected.toSet
      }

      new BitSet {
        def terms = atoms
        override def getWidth: Int = width
        override def toString: String = s"BitSetRange(0x${start.toString(16)} - 0x${(start + length).toString(16)})"
      }
    }
  }

  /** A Set of `BitPat` represents a set of bit vector with mask. */
  sealed trait BitSet { outer =>

    /** all `BitPat` elements in [[terms]] make up this `BitSet`.
      * all [[terms]] should be have the same width.
      */
    def terms: Set[BitPat]

    /** Get specified width of said BitSet */
    def getWidth: Int = {
      require(terms.map(_.width).size <= 1, s"All BitPats must be the same size! Got $this")
      // set width = 0 if terms is empty.
      terms.headOption.map(_.width).getOrElse(0)
    }

    import BitPat.bitPatOrder
    override def toString: String = terms.toSeq.sorted.mkString("\n")

    /** whether this `BitSet` is empty (i.e. no value matches) */
    def isEmpty: Boolean = terms.forall(_.isEmpty)

    def matches(input: UInt) = VecInit(terms.map(_ === input).toSeq).asUInt.orR

    /** Check whether this `BitSet` overlap with that `BitSet`, i.e. !(intersect.isEmpty)
      *
      * @param that `BitSet` to be checked.
      * @return true if this and that `BitSet` have overlap.
      */
    def overlap(that: BitSet): Boolean =
      !terms.flatMap(a => that.terms.map(b => (a, b))).forall { case (a, b) => !a.overlap(b) }

    /** Check whether this `BitSet` covers that (i.e. forall b matches that, b also matches this)
      *
      * @param that `BitSet` to b covered
      * @return true if this `BitSet` can cover that `BitSet`
      */
    def cover(that: BitSet): Boolean =
      that.subtract(this).isEmpty

    /** Intersect `this` and `that` `BitSet`.
      *
      * @param that `BitSet` to be intersected.
      * @return a `BitSet` containing all elements of `this` that also belong to `that`.
      */
    def intersect(that: BitSet): BitSet =
      terms
        .flatMap(a => that.terms.map(b => a.intersect(b)))
        .filterNot(_.isEmpty)
        .fold(BitSet.empty)(_.union(_))

    /** Subtract that from this `BitSet`.
      *
      * @param that subtrahend `BitSet`.
      * @return a `BitSet` contining elements of `this` which are not the elements of `that`.
      */
    def subtract(that: BitSet): BitSet =
      terms.map { a =>
        that.terms.map(b => a.subtract(b)).fold(a)(_.intersect(_))
      }.filterNot(_.isEmpty).fold(BitSet.empty)(_.union(_))

    /** Union this and that `BitSet`
      *
      * @param that `BitSet` to union.
      * @return a `BitSet` containing all elements of `this` and `that`.
      */
    def union(that: BitSet): BitSet = new BitSet {
      def terms = outer.terms ++ that.terms
    }

    /** Test whether two `BitSet` matches the same set of value
      *
      * @note
      * This method can be very expensive compared to ordinary == operator between two Objects
      *
      * @return true if two `BitSet` is same.
      */
    override def equals(obj: Any): Boolean = {
      obj match {
        case that: BitSet => this.getWidth == that.getWidth && this.cover(that) && that.cover(this)
        case _ => false
      }
    }

    /**
      * Calculate the inverse of this pattern set.
      *
      * @return A BitSet matching all value (of the given with) iff it doesn't match this pattern.
      */
    def inverse: BitSet = {
      val total = BitPat("b" + ("?" * this.getWidth))
      total.subtract(this)
    }
  }
}

/** Bit patterns are literals with masks, used to represent values with don't
  * care bits. Equality comparisons will ignore don't care bits.
  *
  * @example {{{
  * "b10101".U === BitPat("b101??") // evaluates to true.B
  * "b10111".U === BitPat("b101??") // evaluates to true.B
  * "b10001".U === BitPat("b101??") // evaluates to false.B
  * }}}
  */
sealed class BitPat(val value: BigInt, val mask: BigInt, val width: Int)
    extends util.experimental.BitSet
    with SourceInfoDoc {
  import chisel3.util.experimental.BitSet
  def terms = Set(this)

  /**
    * Get specified width of said BitPat
    */
  override def getWidth: Int = width
  def apply(x:  Int): BitPat = macro SourceInfoTransform.xArg
  def apply(x:  Int, y: Int): BitPat = macro SourceInfoTransform.xyArg
  def ===(that: UInt):   Bool = macro SourceInfoTransform.thatArg
  def =/=(that: UInt):   Bool = macro SourceInfoTransform.thatArg
  def ##(that:  BitPat): BitPat = macro SourceInfoTransform.thatArg

  override def equals(obj: Any): Boolean = {
    obj match {
      case that: BitPat => this.value == that.value && this.mask == that.mask && this.width == that.width
      case that: BitSet => super.equals(obj)
      case _ => false
    }
  }

  override def hashCode: Int =
    MurmurHash3.seqHash(Seq(this.value, this.mask, this.width))

  /** @group SourceInfoTransformMacro */
  def do_apply(x: Int)(implicit sourceInfo: SourceInfo): BitPat = {
    do_apply(x, x)
  }

  /** @group SourceInfoTransformMacro */
  def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo): BitPat = {
    require(width > x && y >= 0, s"Invalid bit range ($x, $y), index should be bounded by (${width - 1}, 0)")
    require(x >= y, s"Invalid bit range ($x, $y), x should be greater or equal to y.")
    BitPat(s"b${rawString.slice(width - x - 1, width - y)}")
  }

  /** @group SourceInfoTransformMacro */
  def do_===(that: UInt)(implicit sourceInfo: SourceInfo): Bool = {
    value.asUInt === (that & mask.asUInt)
  }

  /** @group SourceInfoTransformMacro */
  def do_=/=(that: UInt)(implicit sourceInfo: SourceInfo): Bool = {
    !(this === that)
  }

  /** @group SourceInfoTransformMacro */
  def do_##(that: BitPat)(implicit sourceInfo: SourceInfo): BitPat = {
    new BitPat((value << that.getWidth) + that.value, (mask << that.getWidth) + that.mask, this.width + that.getWidth)
  }

  /** Check whether this `BitPat` overlap with that `BitPat`, i.e. !(intersect.isEmpty)
    *
    * @param that `BitPat` to be checked.
    * @return true if this and that `BitPat` have overlap.
    */
  override def overlap(that: BitSet): Boolean = that match {
    case that: BitPat => ((mask & that.mask) & (value ^ that.value)) == 0
    case _ => super.overlap(that)
  }

  /** Check whether this `BitSet` covers that (i.e. forall b matches that, b also matches this)
    *
    * @param that `BitPat` to be covered
    * @return true if this `BitSet` can cover that `BitSet`
    */
  override def cover(that: BitSet): Boolean = that match {
    case that: BitPat => (mask & (~that.mask | (value ^ that.value))) == 0
    case _ => super.cover(that)
  }

  /** Intersect `this` and `that` `BitPat`.
    *
    * @param that `BitPat` to be intersected.
    * @return a `BitSet` containing all elements of `this` that also belong to `that`.
    */
  def intersect(that: BitPat): BitSet = {
    if (!overlap(that)) {
      BitSet.empty
    } else {
      new BitPat(this.value | that.value, this.mask | that.mask, this.width.max(that.width))
    }
  }

  /** Subtract a `BitPat` from this.
    *
    * @param that subtrahend `BitPat`.
    * @return a `BitSet` containing elements of `this` which are not the elements of `that`.
    */
  def subtract(that: BitPat): BitSet = {
    require(width == that.width)
    def enumerateBits(mask: BigInt): Seq[BigInt] = {
      if (mask == 0) {
        Nil
      } else {
        // bits comes after the first '1' in a number are inverted in its two's complement.
        // therefore bit is always the first '1' in x (counting from least significant bit).
        val bit = mask & (-mask)
        bit +: enumerateBits(mask & ~bit)
      }
    }

    val intersection = intersect(that)
    val omask = this.mask
    if (intersection.isEmpty) {
      this
    } else {
      new BitSet {
        val terms =
          intersection.terms.flatMap { remove =>
            enumerateBits(~omask & remove.mask).map { bit =>
              // Only care about higher than current bit in remove
              val nmask = (omask | ~(bit - 1)) & remove.mask
              val nvalue = (remove.value ^ bit) & nmask
              val nwidth = remove.width
              new BitPat(nvalue, nmask, nwidth)
            }
          }
      }
    }
  }

  override def isEmpty: Boolean = false

  /** Generate raw string of a `BitPat`. */
  def rawString: String = _rawString

  // This is micro-optimized and memoized because it is used for lots of BitPat operations
  private lazy val _rawString: String = {
    val sb = new StringBuilder(width)
    var i = 0
    while (i < width) {
      val bitIdx = width - i - 1
      val char =
        if (mask.testBit(bitIdx)) {
          if (value.testBit(bitIdx)) {
            '1'
          } else {
            '0'
          }
        } else {
          '?'
        }
      sb += char
      i += 1
    }
    sb.result()
  }

  override def toString = s"BitPat($rawString)"
}
