// See LICENSE for license details.

package Chisel

object BitPat {
  /** Parses a bit pattern string into (bits, mask, width).
    *
    * @return bits the literal value, with don't cares being 0
    * @return mask the mask bits, with don't cares being 0 and cares being 1
    * @return width the number of bits in the literal, including values and
    * don't cares.
    */
  private def parse(x: String): (BigInt, BigInt, Int) = {
    // REVIEW TODO: can this be merged with literal parsing creating one unified
    // Chisel string to value decoder (which can also be invoked by libraries
    // and testbenches?
    // REVIEW TODO: Verilog Xs also handle octal and hex cases.
    require(x.head == 'b', "BitPats must be in binary and be prefixed with 'b'")
    var bits = BigInt(0)
    var mask = BigInt(0)
    for (d <- x.tail) {
      if (d != '_') {
        if (!"01?".contains(d)) Builder.error({"Literal: " + x + " contains illegal character: " + d})
        mask = (mask << 1) + (if (d == '?') 0 else 1)
        bits = (bits << 1) + (if (d == '1') 1 else 0)
      }
    }
    (bits, mask, x.length - 1)
  }

  /** Creates a [[BitPat]] literal from a string.
   *
    * @param n the literal value as a string, in binary, prefixed with 'b'
    * @note legal characters are '0', '1', and '?', as well as '_' as white
    * space (which are ignored)
    */
  def apply(n: String): BitPat = {
    val (bits, mask, width) = parse(n)
    new BitPat(bits, mask, width)
  }

  /** Creates a [[BitPat]] of all don't cares of a specified width. */
  // REVIEW TODO: is this really necessary? if so, can there be a better name?
  def DC(width: Int): BitPat = BitPat("b" + ("?" * width))

  // BitPat <-> UInt
  /** enable conversion of a bit pattern to a UInt */
  // REVIEW TODO: Doesn't having a BitPat with all mask bits high defeat the
  // point of using a BitPat in the first place?
  implicit def BitPatToUInt(x: BitPat): UInt = {
    require(x.mask == (BigInt(1) << x.getWidth) - 1)
    UInt(x.value, x.getWidth)
  }

  /** create a bit pattern from a UInt */
  // REVIEW TODO: Similar, what is the point of this?
  implicit def apply(x: UInt): BitPat = {
    require(x.isLit)
    BitPat("b" + x.litValue.toString(2))
  }
}

// TODO: Break out of Core? (this doesn't involve FIRRTL generation)
/** Bit patterns are literals with masks, used to represent values with don't
  * cares. Equality comparisons will ignore don't care bits (for example,
  * BitPat(0b10?1) === UInt(0b1001) and UInt(0b1011)).
  */
sealed class BitPat(val value: BigInt, val mask: BigInt, width: Int) {
  def getWidth: Int = width
  def === (other: UInt): Bool = UInt(value) === (other & UInt(mask))
  def != (other: UInt): Bool = !(this === other)
}
