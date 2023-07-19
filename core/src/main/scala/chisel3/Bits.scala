// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import chisel3.experimental.{requireIsHardware, SourceInfo}
import chisel3.internal.{throwException, BaseModule}
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{
  IntLiteralApplyTransform,
  SourceInfoTransform,
  SourceInfoWhiteboxTransform,
  UIntTransform
}
import chisel3.internal.firrtl.PrimOp._
import _root_.firrtl.{ir => firrtlir}
import chisel3.internal.{castToInt, Builder, Warning, WarningID}

/** Exists to unify common interfaces of [[Bits]] and [[Reset]].
  *
  * @note This is a workaround because macros cannot override abstract methods.
  */
private[chisel3] sealed trait ToBoolable extends Element {

  /** Casts this $coll to a [[Bool]]
    *
    * @note The width must be known and equal to 1
    */
  final def asBool: Bool = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo): Bool
}

/** A data type for values represented by a single bitvector. This provides basic bitwise operations.
  *
  * @groupdesc Bitwise Bitwise hardware operators
  * @define coll [[Bits]]
  * @define sumWidthInt    @note The width of the returned $coll is `width of this` + `that`.
  * @define sumWidth       @note The width of the returned $coll is `width of this` + `width of that`.
  * @define unchangedWidth @note The width of the returned $coll is unchanged, i.e., the `width of this`.
  */
sealed abstract class Bits(private[chisel3] val width: Width) extends Element with ToBoolable {
  // TODO: perhaps make this concrete?
  // Arguments for: self-checking code (can't do arithmetic on bits)
  // Arguments against: generates down to a FIRRTL UInt anyways

  // Only used for in a few cases, hopefully to be removed
  private[chisel3] def cloneTypeWidth(width: Width): this.type

  def cloneType: this.type = cloneTypeWidth(width)

  /** A non-ambiguous name of this `Bits` instance for use in generated Verilog names
    * Inserts the width directly after the typeName, e.g. UInt4, SInt1
    */
  override def typeName: String = s"${this.getClass.getSimpleName}$width"

  /** Tail operator
    *
    * @param n the number of bits to remove
    * @return This $coll with the `n` most significant bits removed.
    * @group Bitwise
    */
  final def tail(n: Int): UInt = macro SourceInfoTransform.nArg

  /** Head operator
    *
    * @param n the number of bits to take
    * @return The `n` most significant bits of this $coll
    * @group Bitwise
    */
  final def head(n: Int): UInt = macro SourceInfoTransform.nArg

  /** @group SourceInfoTransformMacro */
  def do_tail(n: Int)(implicit sourceInfo: SourceInfo): UInt = {
    val w = width match {
      case KnownWidth(x) =>
        require(x >= n, s"Can't tail($n) for width $x < $n")
        Width(x - n)
      case UnknownWidth() => Width()
    }
    binop(sourceInfo, UInt(width = w), TailOp, n)
  }

  /** @group SourceInfoTransformMacro */
  def do_head(n: Int)(implicit sourceInfo: SourceInfo): UInt = {
    width match {
      case KnownWidth(x)  => require(x >= n, s"Can't head($n) for width $x < $n")
      case UnknownWidth() =>
    }
    binop(sourceInfo, UInt(Width(n)), HeadOp, n)
  }

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def extract(x: BigInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_extract(x: BigInt)(implicit sourceInfo: SourceInfo): Bool = {
    if (x < 0) {
      Builder.error(s"Negative bit indices are illegal (got $x)")
    }
    // This preserves old behavior while a more more consistent API is under debate
    // See https://github.com/freechipsproject/chisel3/issues/867
    litOption.map { value =>
      (((value >> castToInt(x, "Index")) & 1) == 1).asBool
    }.getOrElse {
      requireIsHardware(this, "bits to be indexed")

      widthOption match {
        case Some(w) if w == 0 => Builder.error(s"Cannot extract from zero-width")
        case Some(w) if x >= w => Builder.error(s"High index $x is out of range [0, ${w - 1}]")
        case _                 =>
      }

      pushOp(DefPrim(sourceInfo, Bool(), BitsExtractOp, this.ref, ILit(x), ILit(x)))
    }
  }

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def apply(x: BigInt): Bool = macro IntLiteralApplyTransform.safeApply

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: BigInt)(implicit sourceInfo: SourceInfo): Bool =
    do_extract(x)

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def apply(x: Int): Bool = macro IntLiteralApplyTransform.safeApply

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: Int)(implicit sourceInfo: SourceInfo): Bool =
    do_extract(BigInt(x))

  /** Returns the specified bit on this wire as a [[Bool]], dynamically addressed.
    *
    * @param x a hardware component whose value will be used for dynamic addressing
    * @return the specified bit
    */
  final def extract(x: UInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_extract(x: UInt)(implicit sourceInfo: SourceInfo): Bool = {
    this.widthOption.foreach { thisWidth =>
      if (thisWidth == 0) {
        Builder.error(s"Cannot extract from zero-width")
      } else {
        x.widthOption.foreach { xWidth =>
          if (xWidth >= 31 || (1 << (xWidth - 1)) >= thisWidth) {
            val msg = s"Dynamic index with width $xWidth is too large for extractee of width $thisWidth"
            Builder.warning(Warning(WarningID.DynamicBitSelectTooWide, msg))
          } else if ((1 << xWidth) < thisWidth) {
            val msg = s"Dynamic index with width $xWidth is too small for extractee of width $thisWidth"
            Builder.warning(Warning(WarningID.DynamicBitSelectTooNarrow, msg))
          }
        }
      }
    }
    val theBits = this >> x
    val noExtract = theBits.widthOption.exists(_ <= 1)
    if (noExtract) theBits.asBool else theBits(0)
  }

  /** Returns the specified bit on this wire as a [[Bool]], dynamically addressed.
    *
    * @param x a hardware component whose value will be used for dynamic addressing
    * @return the specified bit
    */
  final def apply(x: UInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: UInt)(implicit sourceInfo: SourceInfo): Bool =
    do_extract(x)

  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    *   val myBits = "0b101".U
    *   myBits(1, 0) // "0b01".U  // extracts the two least significant bits
    *
    *   // Note that zero-width ranges are also legal
    *   myBits(-1, 0) // 0.U(0.W) // zero-width UInt
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component containing the requested bits
    */
  final def apply(x: Int, y: Int): UInt = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo): UInt = {
    if ((x < y && !(x == -1 && y == 0)) || y < 0) {
      val zeroWidthSuggestion =
        if (x == y - 1) {
          s". If you are trying to extract zero-width range, right-shift by 'lo' before extracting."
        } else {
          ""
        }
      Builder.error(s"Invalid bit range [hi=$x, lo=$y]$zeroWidthSuggestion")
    }
    val resultWidth = x - y + 1
    // This preserves old behavior while a more more consistent API is under debate
    // See https://github.com/freechipsproject/chisel3/issues/867
    litOption.map { value =>
      ((value >> y) & ((BigInt(1) << resultWidth) - 1)).asUInt(resultWidth.W)
    }.getOrElse {
      requireIsHardware(this, "bits to be sliced")

      widthOption match {
        case Some(w) if w == 0 => Builder.error(s"Cannot extract from zero-width")
        case Some(w) if y >= w => Builder.error(s"High and low indices $x and $y are both out of range [0, ${w - 1}]")
        case Some(w) if x >= w => Builder.error(s"High index $x is out of range [0, ${w - 1}]")
        case _                 =>
      }

      // FIRRTL does not yet support empty extraction so we must return the zero-width wire here:
      if (resultWidth == 0) {
        0.U(0.W)
      } else {
        pushOp(DefPrim(sourceInfo, UInt(Width(resultWidth)), BitsExtractOp, this.ref, ILit(x), ILit(y)))
      }
    }
  }

  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    *   val myBits = "0b101".U
    *   myBits(1, 0) // "0b01".U  // extracts the two least significant bits
    *
    *   // Note that zero-width ranges are also legal
    *   myBits(-1, 0) // 0.U(0.W) // zero-width UInt
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component containing the requested bits
    */
  final def apply(x: BigInt, y: BigInt): UInt = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    do_apply(castToInt(x, "High index"), castToInt(y, "Low index"))

  private[chisel3] def unop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp): T = {
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref))
  }
  private[chisel3] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: BigInt): T = {
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, ILit(other)))
  }
  private[chisel3] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: Bits): T = {
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, other.ref))
  }
  private[chisel3] def compop(sourceInfo: SourceInfo, op: PrimOp, other: Bits): Bool = {
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }
  private[chisel3] def redop(sourceInfo: SourceInfo, op: PrimOp): Bool = {
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref))
  }

  /** Pad operator
    *
    * @param that the width to pad to
    * @return this @coll zero padded up to width `that`. If `that` is less than the width of the original component,
    * this method returns the original component.
    * @note For [[SInt]]s only, this will do sign extension.
    * @group Bitwise
    */
  final def pad(that: Int): this.type = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_pad(that: Int)(implicit sourceInfo: SourceInfo): this.type = this.width match {
    case KnownWidth(w) if w >= that => this
    case _                          => binop(sourceInfo, cloneTypeWidth(this.width.max(Width(that))), PadOp, that)
  }

  /** Bitwise inversion operator
    *
    * @return this $coll with each bit inverted
    * @group Bitwise
    */
  final def unary_~ : Bits = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_~(implicit sourceInfo: SourceInfo): Bits

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or Bits?
  final def <<(that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<<(that: BigInt)(implicit sourceInfo: SourceInfo): Bits

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  final def <<(that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<<(that: Int)(implicit sourceInfo: SourceInfo): Bits

  /** Dynamic left shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
    * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
    * @group Bitwise
    */
  final def <<(that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<<(that: UInt)(implicit sourceInfo: SourceInfo): Bits

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  // REVIEW TODO: redundant
  final def >>(that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>>(that: BigInt)(implicit sourceInfo: SourceInfo): Bits

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  final def >>(that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>>(that: Int)(implicit sourceInfo: SourceInfo): Bits

  /** Dynamic right shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
    * significant bits.
    * $unchangedWidth
    * @group Bitwise
    */
  final def >>(that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>>(that: UInt)(implicit sourceInfo: SourceInfo): Bits

  /** Returns the contents of this wire as a [[scala.collection.Seq]] of [[Bool]]. */
  final def asBools: Seq[Bool] = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asBools(implicit sourceInfo: SourceInfo): Seq[Bool] =
    Seq.tabulate(this.getWidth)(i => this(i))

  /** Reinterpret this $coll as an [[SInt]]
    *
    * @note The arithmetic value is not preserved if the most-significant bit is set. For example, a [[UInt]] of
    * width 3 and value 7 (0b111) would become an [[SInt]] of width 3 and value -1.
    */
  final def asSInt: SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asSInt(implicit sourceInfo: SourceInfo): SInt

  final def do_asBool(implicit sourceInfo: SourceInfo): Bool = {
    width match {
      case KnownWidth(1) => this(0)
      case _             => throwException(s"can't covert ${this.getClass.getSimpleName}$width to Bool")
    }
  }

  /** Concatenation operator
    *
    * @param that a hardware component
    * @return this $coll concatenated to the most significant end of `that`
    * $sumWidth
    * @group Bitwise
    */
  final def ##(that: Bits): UInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_##(that: Bits)(implicit sourceInfo: SourceInfo): UInt = {
    val w = this.width + that.width
    pushOp(DefPrim(sourceInfo, UInt(w), ConcatOp, this.ref, that.ref))
  }

  /** Default print as [[Decimal]] */
  final def toPrintable: Printable = Decimal(this)

  protected final def validateShiftAmount(x: Int)(implicit sourceInfo: SourceInfo): Int = {
    if (x < 0)
      Builder.error(s"Negative shift amounts are illegal (got $x)")
    x
  }
}

/** A data type for unsigned integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[UInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class UInt private[chisel3] (width: Width) extends Bits(width) with Num[UInt] {
  override def toString: String = {
    litOption match {
      case Some(value) => s"UInt$width($value)"
      case _           => stringAccessor(s"UInt$width")
    }
  }

  private[chisel3] override def cloneTypeWidth(w: Width): this.type =
    new UInt(w).asInstanceOf[this.type]

  // TODO: refactor to share documentation with Num or add independent scaladoc
  /** Unary negation (expanding width)
    *
    * @return a $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_- : UInt = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
    *
    * @return a $coll equal to zero minus this $coll shifted right by one.
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_-% : UInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_-(implicit sourceInfo: SourceInfo): UInt = 0.U - this

  /** @group SourceInfoTransformMacro */
  def do_unary_-%(implicit sourceInfo: SourceInfo): UInt = 0.U -% this

  override def do_+(that: UInt)(implicit sourceInfo: SourceInfo): UInt = this +% that
  override def do_-(that: UInt)(implicit sourceInfo: SourceInfo): UInt = this -% that
  override def do_/(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width), DivideOp, that)
  override def do_%(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.min(that.width)), RemOp, that)
  override def do_*(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width + that.width), TimesOp, that)

  /** Multiplication operator
    *
    * @param that a hardware [[SInt]]
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def *(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_*(that: SInt)(implicit sourceInfo: SourceInfo): SInt = that * this

  /** Addition operator (expanding width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def +&(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def +%(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def -&(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def -%(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+&(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt((this.width.max(that.width)) + 1), AddOp, that)

  /** @group SourceInfoTransformMacro */
  def do_+%(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this +& that).tail(1)

  /** @group SourceInfoTransformMacro */
  def do_-&(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this.subtractAsSInt(that)).asUInt

  /** @group SourceInfoTransformMacro */
  def do_-%(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this.subtractAsSInt(that)).tail(1)

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def &(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def |(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def ^(that: UInt): UInt = macro SourceInfoTransform.thatArg

  //  override def abs: UInt = macro SourceInfoTransform.noArgDummy
  def do_abs(implicit sourceInfo: SourceInfo): UInt = this

  /** @group SourceInfoTransformMacro */
  def do_&(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitAndOp, that)

  /** @group SourceInfoTransformMacro */
  def do_|(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitOrOp, that)

  /** @group SourceInfoTransformMacro */
  def do_^(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitXorOp, that)

  /** @group SourceInfoTransformMacro */
  def do_unary_~(implicit sourceInfo: SourceInfo): UInt =
    unop(sourceInfo, UInt(width = width), BitNotOp)

  // REVIEW TODO: Can these be defined on Bits?
  /** Or reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll or'd together
    * @group Bitwise
    */
  final def orR: Bool = macro SourceInfoTransform.noArg

  /** And reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll and'd together
    * @group Bitwise
    */
  final def andR: Bool = macro SourceInfoTransform.noArg

  /** Exclusive or (xor) reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll xor'd together
    * @group Bitwise
    */
  final def xorR: Bool = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_orR(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, OrReduceOp)

  /** @group SourceInfoTransformMacro */
  def do_andR(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, AndReduceOp)

  /** @group SourceInfoTransformMacro */
  def do_xorR(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, XorReduceOp)

  override def do_<(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessOp, that)
  override def do_>(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterOp, that)
  override def do_<=(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessEqOp, that)
  override def do_>=(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterEqOp, that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  final def =/=(that: UInt): Bool = macro SourceInfoTransform.thatArg

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  final def ===(that: UInt): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_=/=(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, NotEqualOp, that)

  /** @group SourceInfoTransformMacro */
  def do_===(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, EqualOp, that)

  /** Unary not
    *
    * @return a hardware [[Bool]] asserted if this $coll equals zero
    * @group Bitwise
    */
  final def unary_! : Bool = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_!(implicit sourceInfo: SourceInfo): Bool = this === 0.U(1.W)

  override def do_<<(that: Int)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override def do_<<(that: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    this << castToInt(that, "Shift amount")
  override def do_<<(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>>(that: Int)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  override def do_>>(that: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    this >> castToInt(that, "Shift amount")
  override def do_>>(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width), DynamicShiftRightOp, that)

  /**
    * Circular shift to the left
    * @param that number of bits to rotate
    * @return UInt of same width rotated left n bits
    */
  final def rotateLeft(that: Int): UInt = macro SourceInfoWhiteboxTransform.thatArg

  def do_rotateLeft(n: Int)(implicit sourceInfo: SourceInfo): UInt = width match {
    case _ if (n == 0)             => this
    case KnownWidth(w) if (w <= 1) => this
    case KnownWidth(w) if n >= w   => do_rotateLeft(n % w)
    case _ if (n < 0)              => do_rotateRight(-n)
    case _                         => tail(n) ## head(n)
  }

  /**
    * Circular shift to the right
    * @param that number of bits to rotate
    * @return UInt of same width rotated right n bits
    */
  final def rotateRight(that: Int): UInt = macro SourceInfoWhiteboxTransform.thatArg

  def do_rotateRight(n: Int)(implicit sourceInfo: SourceInfo): UInt = width match {
    case _ if (n <= 0)             => do_rotateLeft(-n)
    case KnownWidth(w) if (w <= 1) => this
    case KnownWidth(w) if n >= w   => do_rotateRight(n % w)
    case _                         => this(n - 1, 0) ## (this >> n)
  }

  final def rotateRight(that: UInt): UInt = macro SourceInfoWhiteboxTransform.thatArg

  private def dynamicShift(
    n:           UInt,
    staticShift: (UInt, Int) => UInt
  )(
    implicit sourceInfo: SourceInfo
  ): UInt =
    n.asBools.zipWithIndex.foldLeft(this) {
      case (in, (en, sh)) => Mux(en, staticShift(in, 1 << sh), in)
    }

  def do_rotateRight(n: UInt)(implicit sourceInfo: SourceInfo): UInt =
    dynamicShift(n, _ rotateRight _)

  final def rotateLeft(that: UInt): UInt = macro SourceInfoWhiteboxTransform.thatArg

  def do_rotateLeft(n: UInt)(implicit sourceInfo: SourceInfo): UInt =
    dynamicShift(n, _ rotateLeft _)

  /** Conditionally set or clear a bit
    *
    * @param off a dynamic offset
    * @param dat set if true, clear if false
    * @return a hrdware $coll with bit `off` set or cleared based on the value of `dat`
    * $unchangedWidth
    */
  final def bitSet(off: UInt, dat: Bool): UInt = macro UIntTransform.bitset

  /** @group SourceInfoTransformMacro */
  def do_bitSet(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo): UInt = {
    val bit = 1.U(1.W) << off
    Mux(dat, this | bit, ~(~this | bit))
  }

  // TODO: this eventually will be renamed as toSInt, once the existing toSInt
  // completes its deprecation phase.
  /** Zero extend as [[SInt]]
    *
    * @return an [[SInt]] equal to this $coll with an additional zero in its most significant bit
    * @note The width of the returned [[SInt]] is `width of this` + `1`.
    */
  final def zext: SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_zext(implicit sourceInfo: SourceInfo): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width + 1), ConvertOp, ref))

  override def do_asSInt(implicit sourceInfo: SourceInfo): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width), AsSIntOp, ref))
  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = this

  private[chisel3] override def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    this := that.asUInt
  }

  private def subtractAsSInt(that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width.max(that.width)) + 1), SubOp, that)
}

/** A data type for signed integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[SInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class SInt private[chisel3] (width: Width) extends Bits(width) with Num[SInt] {
  override def toString: String = {
    litOption match {
      case Some(value) => s"SInt$width($value)"
      case _           => stringAccessor(s"SInt$width")
    }
  }

  private[chisel3] override def cloneTypeWidth(w: Width): this.type =
    new SInt(w).asInstanceOf[this.type]

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_- : SInt = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus `this` shifted right by one
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_-% : SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_-(implicit sourceInfo: SourceInfo): SInt = 0.S - this

  /** @group SourceInfoTransformMacro */
  def do_unary_-%(implicit sourceInfo: SourceInfo): SInt = 0.S -% this

  /** add (default - no growth) operator */
  override def do_+(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    this +% that

  /** subtract (default - no growth) operator */
  override def do_-(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    this -% that
  override def do_*(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + that.width), TimesOp, that)
  override def do_/(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + 1), DivideOp, that)
  override def do_%(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width.min(that.width)), RemOp, that)

  /** Multiplication operator
    *
    * @param that a hardware $coll
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def *(that: UInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_*(that: UInt)(implicit sourceInfo: SourceInfo): SInt = {
    val thatToSInt = that.zext
    val result = binop(sourceInfo, SInt(this.width + thatToSInt.width), TimesOp, thatToSInt)
    result.tail(1).asSInt
  }

  /** Addition operator (expanding width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def +&(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  final def +%(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def -&(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  final def -%(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+&(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width.max(that.width)) + 1), AddOp, that)

  /** @group SourceInfoTransformMacro */
  def do_+%(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    (this +& that).tail(1).asSInt

  /** @group SourceInfoTransformMacro */
  def do_-&(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width.max(that.width)) + 1), SubOp, that)

  /** @group SourceInfoTransformMacro */
  def do_-%(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    (this -& that).tail(1).asSInt

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def &(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def |(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def ^(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitAndOp, that).asSInt

  /** @group SourceInfoTransformMacro */
  def do_|(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitOrOp, that).asSInt

  /** @group SourceInfoTransformMacro */
  def do_^(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitXorOp, that).asSInt

  /** @group SourceInfoTransformMacro */
  def do_unary_~(implicit sourceInfo: SourceInfo): SInt =
    unop(sourceInfo, UInt(width = width), BitNotOp).asSInt

  override def do_<(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessOp, that)
  override def do_>(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterOp, that)
  override def do_<=(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessEqOp, that)
  override def do_>=(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterEqOp, that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  final def =/=(that: SInt): Bool = macro SourceInfoTransform.thatArg

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  final def ===(that: SInt): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_=/=(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, NotEqualOp, that)

  /** @group SourceInfoTransformMacro */
  def do_===(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, EqualOp, that)

//  final def abs(): UInt = macro SourceInfoTransform.noArgDummy

  def do_abs(implicit sourceInfo: SourceInfo): SInt = {
    Mux(this < 0.S, -this, this)
  }

  override def do_<<(that: Int)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override def do_<<(that: BigInt)(implicit sourceInfo: SourceInfo): SInt =
    this << castToInt(that, "Shift amount")
  override def do_<<(that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>>(that: Int)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  override def do_>>(that: BigInt)(implicit sourceInfo: SourceInfo): SInt =
    this >> castToInt(that, "Shift amount")
  override def do_>>(that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width), DynamicShiftRightOp, that)

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = pushOp(
    DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref)
  )
  override def do_asSInt(implicit sourceInfo: SourceInfo): SInt = this

  private[chisel3] override def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    this := that.asSInt
  }
}

sealed trait Reset extends Element with ToBoolable {

  /** Casts this $coll to an [[AsyncReset]] */
  final def asAsyncReset: AsyncReset = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset
}

object Reset {
  def apply(): Reset = new ResetType
}

/** "Abstract" Reset Type inferred in FIRRTL to either [[AsyncReset]] or [[Bool]]
  *
  * @note This shares a common interface with [[AsyncReset]] and [[Bool]] but is not their actual
  * super type due to Bool inheriting from abstract class UInt
  */
final class ResetType(private[chisel3] val width: Width = Width(1)) extends Element with Reset {
  override def toString: String = stringAccessor("Reset")

  def cloneType: this.type = Reset().asInstanceOf[this.type]

  override def litOption = None

  /** Not really supported */
  def toPrintable: Printable = PString("Reset")

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = pushOp(
    DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref)
  )

  private[chisel3] override def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    this := that
  }

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset =
    pushOp(DefPrim(sourceInfo, AsyncReset(), AsAsyncResetOp, ref))

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), AsUIntOp, ref))

  /** @group SourceInfoTransformMacro */
  def do_toBool(implicit sourceInfo: SourceInfo): Bool = do_asBool
}

object AsyncReset {
  def apply(): AsyncReset = new AsyncReset
}

/** Data type representing asynchronous reset signals
  *
  * These signals are similar to [[Clock]]s in that they must be glitch-free for proper circuit
  * operation. [[Reg]]s defined with the implicit reset being an [[AsyncReset]] will be
  * asychronously reset registers.
  */
sealed class AsyncReset(private[chisel3] val width: Width = Width(1)) extends Element with Reset {
  override def toString: String = stringAccessor("AsyncReset")

  def cloneType: this.type = AsyncReset().asInstanceOf[this.type]

  override def litOption = None

  /** Not really supported */
  def toPrintable: Printable = PString("AsyncReset")

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = pushOp(
    DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref)
  )

  // TODO Is this right?
  private[chisel3] override def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    this := that.asBool.asAsyncReset
  }

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset = this

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), AsUIntOp, ref))

  /** @group SourceInfoTransformMacro */
  def do_toBool(implicit sourceInfo: SourceInfo): Bool = do_asBool
}

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  *
  * @define coll [[Bool]]
  * @define numType $coll
  */
sealed class Bool() extends UInt(1.W) with Reset {

  /**
    * Give this `Bool` a stable `typeName` for Verilog name generation.
    * Specifying a Bool's width in its type name isn't necessary
    */
  override def typeName = "Bool"

  override def toString: String = {
    litToBooleanOption match {
      case Some(value) => s"Bool($value)"
      case _           => stringAccessor("Bool")
    }
  }

  private[chisel3] override def cloneTypeWidth(w: Width): this.type = {
    require(!w.known || w.get == 1)
    new Bool().asInstanceOf[this.type]
  }

  /** Convert to a [[scala.Option]] of [[scala.Boolean]] */
  def litToBooleanOption: Option[Boolean] = litOption.map {
    case intVal if intVal == 1 => true
    case intVal if intVal == 0 => false
    case intVal                => throwException(s"Boolean with unexpected literal value $intVal")
  }

  /** Convert to a [[scala.Boolean]] */
  def litToBoolean: Boolean = litToBooleanOption.get

  // REVIEW TODO: Why does this need to exist and have different conventions
  // than Bits?

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * @group Bitwise
    */
  final def &(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * @group Bitwise
    */
  final def |(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * @group Bitwise
    */
  final def ^(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&(that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitAndOp, that)

  /** @group SourceInfoTransformMacro */
  def do_|(that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitOrOp, that)

  /** @group SourceInfoTransformMacro */
  def do_^(that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitXorOp, that)

  /** @group SourceInfoTransformMacro */
  override def do_unary_~(implicit sourceInfo: SourceInfo): Bool =
    unop(sourceInfo, Bool(), BitNotOp)

  /** Logical or operator
    *
    * @param that a hardware $coll
    * @return the logical or of this $coll and `that`
    * @note this is equivalent to [[Bool!.|(that:chisel3\.Bool)* Bool.|)]]
    * @group Logical
    */
  def ||(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_||(that: Bool)(implicit sourceInfo: SourceInfo): Bool = this | that

  /** Logical and operator
    *
    * @param that a hardware $coll
    * @return the logical and of this $coll and `that`
    * @note this is equivalent to [[Bool!.&(that:chisel3\.Bool)* Bool.&]]
    * @group Logical
    */
  def &&(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&&(that: Bool)(implicit sourceInfo: SourceInfo): Bool = this & that

  /** Reinterprets this $coll as a clock */
  def asClock: Clock = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asClock(implicit sourceInfo: SourceInfo): Clock = pushOp(
    DefPrim(sourceInfo, Clock(), AsClockOp, ref)
  )

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset =
    pushOp(DefPrim(sourceInfo, AsyncReset(), AsAsyncResetOp, ref))
}
