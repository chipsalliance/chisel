// SPDX-License-Identifier: Apache-2.0

/** Circuit-land math operations.
  */

package chisel3.util

import chisel3._
import chisel3.internal.{Builder, Warning, WarningID}
import chisel3.experimental.SourceInfo

/** Returns the base-2 integer logarithm of an UInt.
  *
  * @note The result is truncated, so e.g. Log2(13.U) === 3.U
  *
  * @example {{{
  * Log2(8.U)  // evaluates to 3.U
  * Log2(13.U)  // evaluates to 3.U (truncation)
  * Log2(myUIntWire)
  * }}}
  */
object Log2 {

  /** Returns the base-2 integer logarithm of the least-significant `width` bits of an UInt.
    */
  def apply(x: Bits, width: Int): UInt = {
    if (width < 2) {
      0.U
    } else if (width == 2) {
      x(1)
    } else if (width <= divideAndConquerThreshold) {
      Mux(x(width - 1), (width - 1).asUInt, apply(x, width - 1))
    } else {
      val mid = 1 << (log2Ceil(width) - 1)
      val hi = x(width - 1, mid)
      val lo = x(mid - 1, 0)
      val useHi = hi.orR
      Cat(useHi, Mux(useHi, Log2(hi, width - mid), Log2(lo, mid)))
    }
  }

  def apply(x: Bits): UInt = apply(x, x.getWidth)

  private def divideAndConquerThreshold = 4
}

/** Carry-save adder circuit generation functions.
 *  Constructs a tree to reduce an arbitrary number of input terms to two terms.
 *
 *  Example resulting circuit topology if applied to 8 4-bit-wide terms, a...h:
 *  Rank 1: create groups of 3 terms, reducing each group to 2 terms; i,j,k,l,g,h
 *       GROUP0    GROUP1   GROUP2
 *         aaaa      dddd     gggg
 *         bbbb      eeee     hhhh
 *         cccc      ffff
 *        _____+    _____+   _____+
 *         iiii      kkkk     gggg
 *        jjjj.     llll.     hhhh
 *  Rank 2: create groups of 3 terms, reducing each group to 2 terms: m,n,o,p
 *       GROUP0    GROUP1
 *         iiii    llll.
 *        jjjj.     gggg
 *         kkkk     hhhh
 *       ______+  ______+
 *        mmmmm    ooooo
 *        nnnn.    pppp.
 *  Rank 3: create groups of 3 terms, reducing each group to 2 terms: q,r,p
 *       GROUP0   GROUP1
 *        mmmmm    pppp.
 *        nnnn.
 *        ooooo
 *       ______+  ______+
 *        qqqqq    pppp.
 *       rrrrr.
 *  Rank 4: reduce last group of 3 terms to 2 terms: s,t
 *        qqqqq
 *       rrrrr.
 *        pppp.
 *       ______+
 *       ssssss
 *       tttt..
 *
 * Every . is an empty spot introduced by the topology, constant zero in the carry LSB.
 * If the input has single-bit terms, insert those there.
 * In the example tree above, there are six spots where we can insert a single-bit term: j,l,n,p,r,t
 *
 * Future improvement suggestion (not implemented yet):
 * - add support for negative weights at arbitrary bit positions; required to support subtraction.
 * - circuitry returns two UInt hardware terms; final addition includes a signed constant offset.
 */

object Csa {

  /** Adds an arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  The bits from the *bits parameter are inserted in the tree with LSB weight, without adding logic depth.
   */
  def apply(terms: Seq[UInt], bits: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt = {
    val sumRedundant = sumCarry(terms, bits)
    terms.length match {
      case 0 => 0.U(0.W)
      case 1 => terms.head // avoid +& because it widens result by 1 bit
      case _ => sumRedundant._1 +& sumRedundant._2 // final carry-propagate addition of the two output terms
    }
  }

  /** Adds the arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   */
  def apply(terms: Seq[UInt])(implicit sourceInfo: SourceInfo): UInt = apply(terms, Seq.empty[Bool])

  /** Adds an arbitrary number of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   */
  def apply(firstTerm: UInt, moreTerms: UInt*)(implicit sourceInfo: SourceInfo): UInt = apply(firstTerm +: moreTerms)

  /** Adds the arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  Inserts an additional bit in the LSB bit position, without adding logic depth.
   */
  def apply(terms: Seq[UInt], bit: Bool)(implicit sourceInfo: SourceInfo): UInt = apply(terms, Seq(bit))

  /** Adds the arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  Inserts additional bits in the LSB bit position, without adding logic depth.
   */
  def apply(terms: Seq[UInt], bit: Bool, moreBits: Bool*)(implicit sourceInfo: SourceInfo): UInt =
    apply(terms, bit +: moreBits)

  /** Adds the arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  The bits from the *bits parameter are inserted in the tree with LSB weight, without adding logic depth.
   *  Returns the sum in redundant format as sum/carry tuple.
   */
  def sumCarry(terms: Seq[UInt], bits: Seq[Bool])(implicit sourceInfo: SourceInfo): (UInt, UInt) = {
    val bitsIt = bits.iterator
    val result = carrySaveRec(terms, bitsIt)
    require(
      !bitsIt.hasNext,
      "Not enough 3:2 reduction stages to accommodate all single-bit terms from the second argument"
    )
    terms.filterNot(_.isWidthKnown).foreach { t =>
      Builder.warning(
        Warning(
          WarningID.CsaUnknownInputWidth,
          s"Cannot optimize width of carry vector because width of input term ${t} is unknown."
        )
      )
    }
    result.length match { // Recursive function returns a Seq, convert to tuple
      case 0 => (0.U(0.W), 0.U(0.W))
      case 1 => (result.head, 0.U(0.W))
      case 2 => (result.head, result.last)
    }
  }

  /** Adds the arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  Returns the sum in redundant format as sum/carry tuple.
   */
  def sumCarry(terms: Seq[UInt])(implicit sourceInfo: SourceInfo): (UInt, UInt) = sumCarry(terms, Seq.empty[Bool])

  /** Adds an arbitrary number of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  Returns the sum in redundant format as sum/carry tuple.
   */
  def sumCarry(firstTerm: UInt, moreTerms: UInt*)(implicit sourceInfo: SourceInfo): (UInt, UInt) =
    sumCarry(firstTerm +: moreTerms, Seq.empty[Bool])

  /** Adds the arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  Inserts an additional bit in the LSB bit position, without adding logic depth.
   *  Returns the sum in redundant format as sum/carry tuple.
   */
  def sumCarry(terms: Seq[UInt], bit: Bool)(implicit sourceInfo: SourceInfo): (UInt, UInt) = sumCarry(terms, Seq(bit))

  /** Adds the arbitrary-length sequence of UInts in an area- and timing-efficient way, by using a Carry-Save Adder tree.
   *  Inserts additional bits in the LSB bit position, without adding logic depth.
   *  Returns the sum in redundant format as sum/carry tuple.
   */
  def sumCarry(terms: Seq[UInt], bit: Bool, moreBits: Bool*)(implicit sourceInfo: SourceInfo): (UInt, UInt) =
    sumCarry(terms, bit +: moreBits)

  /** Recursive function to construct a carry save adder tree (Wallace tree). Carry LSB holes are filled with Bools, if provided
   *  Sum is returned as two UInts, sum and carry. These terms are still to be added by a carry-propagate adder.
   */
  private def carrySaveRec(terms: Seq[UInt], bitsIt: Iterator[Bool]): Seq[UInt] = {
    terms.length match {
      case 0 => Seq(0.U(0.W))
      case 1 => terms
      case 2 => terms
      case 3 =>
        val sum = terms(0) ^ terms(1) ^ terms(2)
        val carry = (terms(0) & terms(1) | terms(0) & terms(2) | terms(1) & terms(2)) << 1
        val carryLsb = bitsIt.nextOption().getOrElse(false.B)
        if (terms.forall(_.isWidthKnown)) {
          val carryWidth = terms.map(_.getWidth).sorted.apply(1) + 1
          Seq(sum, carry(carryWidth - 1, 0) | carryLsb)
        } else
          Seq(sum, carry | carryLsb)
      case _ =>
        // Create groups of 3, reduce every group to 2. Result to next level.
        carrySaveRec(terms.grouped(3).map(xyz => carrySaveRec(xyz, bitsIt)).reduce(_ ++ _), bitsIt)
    }
  }
}
