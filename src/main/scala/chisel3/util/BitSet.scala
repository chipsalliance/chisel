// SPDX-License-Identifier: Apache-2.0

package chisel3.util

trait BitSet { bsf =>
  val terms: Set[BitPat]

  /**
    * Get specified width of said BitSet
    *
    * All BitPat contained in one set should have the same width
    */
  def getWidth: Int = {
    assert(terms.map(_.width).size <= 1)
    // set width = 0 if terms is empty.
    terms.headOption.map(_.width).getOrElse(0)
  }

  override def toString: String = terms.toSeq.sortBy((t: BitPat) => (t.mask, t.value)).mkString("\n")

  /**
    * @return whether the BitSetFamily is empty (i.e. no value matches)
    */
  def isEmpty: Boolean = terms.forall(_.isEmpty)

  /**
    * @return whether this BitSetFamily overlap with that BitSetFamily, i.e. !(intersect.isEmpty)
    */
  def overlap(that: BitSet): Boolean = {
    !bsf.terms.flatMap(a => that.terms.map(b => (a, b))).forall { case (a, b) => !a.overlap(b) }
  }

  /**
    * @param that BitSetFamily to be covered
    * @return whether this BitSetFamily covers that (i.e. forall b matches that, b also matches this)
    */
  def cover(that: BitSet): Boolean =
    that.subtract(this).isEmpty

  /**
    * @return a BitSetFamily that only match a value when both operand match
    */
  def intersect(that: BitSet): BitSet =
    bsf.terms
      .flatMap(a => that.terms.map(b => a.intersect(b)))
      .filterNot(_.isEmpty)
      .fold(BitSet.emptyBitSet)(_.union(_))

  /**
    * Subtract that from this BitSetFamily
    * @param that subtrahend
    */
  def subtract(that: BitSet): BitSet =
    bsf.terms.map { a =>
      that.terms.map(b => a.subtract(b)).fold(a)(_.intersect(_))
    }.filterNot(_.isEmpty).fold(BitSet.emptyBitSet)(_.union(_))

  /**
    * Union of two BitSetFamily
    */
  def union(that: BitSet): BitSet = new BitSet {
    val terms = bsf.terms ++ that.terms
  }

  /**
    * Test whether two BitSetFamily matches the same set of value
    *
    * Caution: This method can be very expensive compared to ordinary == operator between two Objects
    */
  override def equals(obj: Any): Boolean = {
    obj match {
      case that: BitSet => if (this.getWidth == that.getWidth) this.cover(that) && that.cover(this) else false
      case _ => false
    }
  }
}

object BitSet {
  val emptyBitSet: BitSet = new BitSet {
    override val terms = Set()
  }

  def apply(str: String): BitSet = {
    new BitSet {
      val terms = str
        .split('\n')
        .map(str => BitPat(str))
        .toSet
      assert(terms.map(_.width).size <= 1)
    }
  }

  /**
    * Generate a decoder circuit that matches the input to each bitSet.
    *
    * The resulting circuit functions like the following but is optimized with a logic minifier
    * when(input === bitSets(0)) { output := b000001 }
    * .elsewhen (input === bitSets(1)) { output := b000010 }
    * ....
    * .otherwise { if (errorBit) output := b100000 else output := DontCare }
    *
    * @param input input to the decoder circuit, width should be equal to bitSets.width
    * @param bitSets set of ports to be matched, all width should be the equal
    * @param errorBit whether generate an additional decode error bit
    */
  def decode(input: chisel3.UInt, bitSets: Seq[BitSet], errorBit: Boolean = false): chisel3.UInt =
    chisel3.util.experimental.decode.decoder(
      input,
      chisel3.util.experimental.decode.TruthTable(
        {
          bitSets.zipWithIndex.flatMap {
            case (family, i) =>
              family.terms.map(bs =>
                s"${bs.toString.replace("-", "?")}->${if (errorBit) "0"}${"0" * (bitSets.size - i - 1)}1${"0" * i}"
              )
          } ++ Seq(s"${if (errorBit) "1"}${"?" * bitSets.size}")
        }.mkString("\n")
      )
    )
}
