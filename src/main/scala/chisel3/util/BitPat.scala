// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import scala.language.experimental.macros
import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform}


object BitPat {

  private[chisel3] implicit val bitPatOrder = new Ordering[BitPat] {
    import scala.math.Ordered.orderingToOrdered
    def compare(x: BitPat, y: BitPat): Int = (x.getWidth, x.value, x.mask) compare (y.getWidth, y.value, y.mask)
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
      if (! (d == '_' || d.isWhitespace)) {
        require("01?".contains(d), "Literal: " + x + " contains illegal character: " + d)
        mask = (mask << 1) + (if (d == '?') 0 else 1)
        bits = (bits << 1) + (if (d == '1') 1 else 0)
        count += 1
      }
    }

    (bits, mask, count)
  }

  /** Creates a [[BitPat]] literal from a string.
    *
    * @param n the literal value as a string, in binary, prefixed with 'b'
    * @note legal characters are '0', '1', and '?', as well as '_' and white
    * space (which are ignored)
    */
  def apply(n: String): BitPat = {
    val (bits, mask, width) = parse(n)
    new BitPat(bits, mask, width)
  }

  /** Creates a [[BitPat]] of all don't cares of the specified bitwidth.
    *
    * @example {{{
    * val myDontCare = BitPat.dontCare(4)  // equivalent to BitPat("b????")
    * }}}
    */
  def dontCare(width: Int): BitPat = BitPat("b" + ("?" * width))

  /** Creates a [[BitPat]] of all 1 of the specified bitwidth.
    *
    * @example {{{
    * val myY = BitPat.Y(4)  // equivalent to BitPat("b1111")
    * }}}
    */
  def Y(width: Int = 1): BitPat = BitPat("b" + ("1" * width))

  /** Creates a [[BitPat]] of all 0 of the specified bitwidth.
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
    import internal.sourceinfo.{SourceInfo, SourceInfoTransform}

    import scala.language.experimental.macros

    final def === (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def =/= (that: BitPat): Bool = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_=== (that: BitPat)
               (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = that === x
    /** @group SourceInfoTransformMacro */
    def do_=/= (that: BitPat)
               (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = that =/= x
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
sealed class BitPat(val value: BigInt, val mask: BigInt, width: Int) extends SourceInfoDoc {
  def getWidth: Int = width
  def apply(x: Int): BitPat = macro SourceInfoTransform.xArg
  def apply(x: Int, y: Int): BitPat = macro SourceInfoTransform.xyArg
  def === (that: UInt): Bool = macro SourceInfoTransform.thatArg
  def =/= (that: UInt): Bool = macro SourceInfoTransform.thatArg
  def ## (that: BitPat): BitPat = macro SourceInfoTransform.thatArg
  override def equals(obj: Any): Boolean = {
    obj match {
      case y: BitPat => value == y.value && mask == y.mask && getWidth == y.getWidth
      case _ => false
    }
  }

  /** @group SourceInfoTransformMacro */
  def do_apply(x: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): BitPat = {
    do_apply(x, x)
  }

  /** @group SourceInfoTransformMacro */
  def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): BitPat = {
    require(width > x && y >= 0, s"Invalid bit range ($x, $y), index should be bounded by (${width - 1}, 0)")
    require(x >= y, s"Invalid bit range ($x, $y), x should be greater or equal to y.")
    BitPat(s"b${rawString.slice(width - x - 1, width - y)}")
  }

  /** @group SourceInfoTransformMacro */
  def do_=== (that: UInt)
      (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    value.asUInt === (that & mask.asUInt)
  }
  /** @group SourceInfoTransformMacro */
  def do_=/= (that: UInt)
      (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    !(this === that)
  }
  /** @group SourceInfoTransformMacro */
  def do_##(that: BitPat)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): BitPat = {
    new BitPat((value << that.getWidth) + that.value, (mask << that.getWidth) + that.mask, this.width + that.getWidth)
  }

  /** Generate raw string of a BitPat. */
  def rawString: String = Seq.tabulate(width) { i =>
      (value.testBit(width - i - 1), mask.testBit(width - i - 1)) match {
      case (true, true) => "1"
      case (false, true) => "0"
      case (_, false) => "?"
    }
  }.mkString

  override def toString = s"BitPat($rawString)"
}
