// See LICENSE for license details.

package Chisel

import scala.language.experimental.macros

import Chisel.internal.sourceinfo.{SourceInfo, SourceInfoTransform}

object BitPat {
  /** Parses a bit pattern string into (bits, mask, width).
    *
    * @return bits the literal value, with don't cares being 0
    * @return mask the mask bits, with don't cares being 0 and cares being 1
    * @return width the number of bits in the literal, including values and
    * don't cares.
    */
  private def parse(x: String): (BigInt, BigInt, Int) = {
    // Notes:
    // While Verilog Xs also handle octal and hex cases, there isn't a
    // compelling argument and no one has asked for it.
    // If ? parsing is to be exposed, the return API needs further scrutiny
    // (especially with things like mask polarity).
    require(x.head == 'b', "BitPats must be in binary and be prefixed with 'b'")
    var bits = BigInt(0)
    var mask = BigInt(0)
    for (d <- x.tail) {
      if (d != '_') {
        require("01?".contains(d), "Literal: " + x + " contains illegal character: " + d)
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

  /** Creates a [[BitPat]] of all don't cares of the specified bitwidth. */
  def dontCare(width: Int): BitPat = BitPat("b" + ("?" * width))

  @deprecated("Use BitPat.dontCare", "chisel3")
  def DC(width: Int): BitPat = dontCare(width)  // scalastyle:ignore method.name

  /** Allows BitPats to be used where a UInt is expected.
    *
    * @note the BitPat must not have don't care bits (will error out otherwise)
    */
  implicit def bitPatToUInt(x: BitPat): UInt = {
    require(x.mask == (BigInt(1) << x.getWidth) - 1)
    UInt(x.value, x.getWidth)
  }

  /** Allows UInts to be used where a BitPat is expected, useful for when an
    * interface is defined with BitPats but not all cases need the partial
    * matching capability.
    *
    * @note the UInt must be a literal
    */
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
  def === (that: UInt): Bool = macro SourceInfoTransform.thatArg
  def =/= (that: UInt): Bool = macro SourceInfoTransform.thatArg
  def != (that: UInt): Bool = macro SourceInfoTransform.thatArg

  def do_=== (that: UInt)(implicit sourceInfo: SourceInfo): Bool = UInt(value) === (that & UInt(mask))
  def do_=/= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = !(this === that)
  def do_!= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = this =/= that
}
