// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import chisel3.experimental.{FixedPoint, Interval}
import chisel3.internal._
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform,
  UIntTransform}
import chisel3.internal.firrtl.PrimOp._
import _root_.firrtl.{ir => firrtlir}
import _root_.firrtl.{constraint => firrtlconstraint}


/** Exists to unify common interfaces of [[Bits]] and [[Reset]].
  *
  * @note This is a workaround because macros cannot override abstract methods.
  */
private[chisel3] sealed trait ToBoolable extends Element {

  /** Casts this $coll to a [[Bool]]
    *
    * @note The width must be known and equal to 1
    */
  final def asBool(): Bool = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool
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
  def do_tail(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val w = width match {
      case KnownWidth(x) =>
        require(x >= n, s"Can't tail($n) for width $x < $n")
        Width(x - n)
      case UnknownWidth() => Width()
    }
    binop(sourceInfo, UInt(width = w), TailOp, n)
  }

  /** @group SourceInfoTransformMacro */
  def do_head(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    width match {
      case KnownWidth(x) => require(x >= n, s"Can't head($n) for width $x < $n")
      case UnknownWidth() =>
    }
    binop(sourceInfo, UInt(Width(n)), HeadOp, n)
  }

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def apply(x: BigInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    if (x < 0) {
      Builder.error(s"Negative bit indices are illegal (got $x)")
    }
    // This preserves old behavior while a more more consistent API is under debate
    // See https://github.com/freechipsproject/chisel3/issues/867
    litOption.map { value =>
      (((value >> castToInt(x, "Index")) & 1) == 1).asBool
    }.getOrElse {
      requireIsHardware(this, "bits to be indexed")
      pushOp(DefPrim(sourceInfo, Bool(), BitsExtractOp, this.ref, ILit(x), ILit(x)))
    }
  }

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    * @note convenience method allowing direct use of [[scala.Int]] without implicits
    */
  final def apply(x: Int): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = apply(BigInt(x))

  /** Returns the specified bit on this wire as a [[Bool]], dynamically addressed.
    *
    * @param x a hardware component whose value will be used for dynamic addressing
    * @return the specified bit
    */
  final def apply(x: UInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    val theBits = this >> x
    theBits(0)
  }

  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    * myBits = 0x5 = 0b101
    * myBits(1,0) => 0b01  // extracts the two least significant bits
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component contain the requested bits
    */
  final def apply(x: Int, y: Int): UInt = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    if (x < y || y < 0) {
      Builder.error(s"Invalid bit range ($x,$y)")
    }
    val w = x - y + 1
    // This preserves old behavior while a more more consistent API is under debate
    // See https://github.com/freechipsproject/chisel3/issues/867
    litOption.map { value =>
      ((value >> y) & ((BigInt(1) << w) - 1)).asUInt(w.W)
    }.getOrElse {
      requireIsHardware(this, "bits to be sliced")
      pushOp(DefPrim(sourceInfo, UInt(Width(w)), BitsExtractOp, this.ref, ILit(x), ILit(y)))
    }
  }

  // REVIEW TODO: again, is this necessary? Or just have this and use implicits?
  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    * myBits = 0x5 = 0b101
    * myBits(1,0) => 0b01  // extracts the two least significant bits
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component contain the requested bits
    */
  final def apply(x: BigInt, y: BigInt): UInt = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    apply(castToInt(x, "High index"), castToInt(y, "Low index"))

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
  def do_pad(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = this.width match {
    case KnownWidth(w) if w >= that => this
    case _ => binop(sourceInfo, cloneTypeWidth(this.width max Width(that)), PadOp, that)
  }

  /** Bitwise inversion operator
    *
    * @return this $coll with each bit inverted
    * @group Bitwise
    */
  final def unary_~ (): Bits = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or Bits?
  final def << (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  final def << (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Dynamic left shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
    * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
    * @group Bitwise
    */
  final def << (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  // REVIEW TODO: redundant
  final def >> (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  final def >> (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Dynamic right shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
    * significant bits.
    * $unchangedWidth
    * @group Bitwise
    */
  final def >> (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Returns the contents of this wire as a [[scala.collection.Seq]] of [[Bool]]. */
  final def toBools(): Seq[Bool] = macro SourceInfoTransform.noArg

  /** Returns the contents of this wire as a [[scala.collection.Seq]] of [[Bool]]. */
  final def asBools(): Seq[Bool] = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asBools(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Seq[Bool] =
    Seq.tabulate(this.getWidth)(i => this(i))

  /** Reinterpret this $coll as an [[SInt]]
    *
    * @note The arithmetic value is not preserved if the most-significant bit is set. For example, a [[UInt]] of
    * width 3 and value 7 (0b111) would become an [[SInt]] of width 3 and value -1.
    */
  final def asSInt(): SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

  /** Reinterpret this $coll as a [[FixedPoint]].
    *
    * @note The value is not guaranteed to be preserved. For example, a [[UInt]] of width 3 and value 7 (0b111) would
    * become a [[FixedPoint]] with value -1. The interpretation of the number is also affected by the specified binary
    * point. '''Caution is advised!'''
    */
  final def asFixedPoint(that: BinaryPoint): FixedPoint = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_asFixedPoint(that: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    throwException(s"Cannot call .asFixedPoint on $this")
  }

  /** Reinterpret cast as a Interval.
    *
    * @note value not guaranteed to be preserved: for example, an UInt of width
    * 3 and value 7 (0b111) would become a FixedInt with value -1, the interpretation
    * of the number is also affected by the specified binary point.  Caution advised
    */
  final def asInterval(that: IntervalRange): Interval = macro SourceInfoTransform.thatArg

  def do_asInterval(that: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    throwException(s"Cannot call .asInterval on $this")
  }

  final def do_asBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    width match {
      case KnownWidth(1) => this(0)
      case _ => throwException(s"can't covert ${this.getClass.getSimpleName}$width to Bool")
    }
  }

  /** Concatenation operator
    *
    * @param that a hardware component
    * @return this $coll concatenated to the most significant end of `that`
    * $sumWidth
    * @group Bitwise
    */
  final def ## (that: Bits): UInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_## (that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val w = this.width + that.width
    pushOp(DefPrim(sourceInfo, UInt(w), ConcatOp, this.ref, that.ref))
  }

  /** Default print as [[Decimal]] */
  final def toPrintable: Printable = Decimal(this)

  protected final def validateShiftAmount(x: Int): Int = {
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
    val bindingString = litOption match {
      case Some(value) => s"($value)"
      case _ => bindingToString
    }
    s"UInt$width$bindingString"
  }

  private[chisel3] override def typeEquivalent(that: Data): Boolean =
    that.isInstanceOf[UInt] && this.width == that.width

  private[chisel3] override def cloneTypeWidth(w: Width): this.type =
    new UInt(w).asInstanceOf[this.type]

  // TODO: refactor to share documentation with Num or add independent scaladoc
  /** Unary negation (expanding width)
    *
    * @return a $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_- (): UInt = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
    *
    * @return a $coll equal to zero minus this $coll shifted right by one.
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_-% (): UInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : UInt = 0.U - this
  /** @group SourceInfoTransformMacro */
  def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = 0.U -% this

  override def do_+ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this +% that
  override def do_- (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this -% that
  override def do_/ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), DivideOp, that)
  override def do_% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width min that.width), RemOp, that)
  override def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width + that.width), TimesOp, that)

  /** Multiplication operator
    *
    * @param that a hardware [[SInt]]
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def * (that: SInt): SInt = macro SourceInfoTransform.thatArg
  /** @group SourceInfoTransformMacro */
  def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = that * this

  /** Addition operator (expanding width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def +& (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def +% (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def -& (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def -% (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt((this.width max that.width) + 1), AddOp, that)
  /** @group SourceInfoTransformMacro */
  def do_+% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this +& that).tail(1)
  /** @group SourceInfoTransformMacro */
  def do_-& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this subtractAsSInt that).asUInt
  /** @group SourceInfoTransformMacro */
  def do_-% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this subtractAsSInt that).tail(1)

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def & (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def | (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def ^ (that: UInt): UInt = macro SourceInfoTransform.thatArg

  //  override def abs: UInt = macro SourceInfoTransform.noArg
  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this

  /** @group SourceInfoTransformMacro */
  def do_& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that)
  /** @group SourceInfoTransformMacro */
  def do_| (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that)
  /** @group SourceInfoTransformMacro */
  def do_^ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that)

  /** @group SourceInfoTransformMacro */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    unop(sourceInfo, UInt(width = width), BitNotOp)

  // REVIEW TODO: Can these be defined on Bits?
  /** Or reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll or'd together
    * @group Bitwise
    */
  final def orR(): Bool = macro SourceInfoTransform.noArg

  /** And reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll and'd together
    * @group Bitwise
    */
  final def andR(): Bool = macro SourceInfoTransform.noArg

  /** Exclusive or (xor) reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll xor'd together
    * @group Bitwise
    */
  final def xorR(): Bool = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_orR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = redop(sourceInfo, OrReduceOp)
  /** @group SourceInfoTransformMacro */
  def do_andR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = redop(sourceInfo, AndReduceOp)
  /** @group SourceInfoTransformMacro */
  def do_xorR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = redop(sourceInfo, XorReduceOp)

  override def do_< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  final def =/= (that: UInt): Bool = macro SourceInfoTransform.thatArg

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  final def === (that: UInt): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_=/= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  /** @group SourceInfoTransformMacro */
  def do_=== (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)

  /** Unary not
    *
    * @return a hardware [[Bool]] asserted if this $coll equals zero
    * @group Bitwise
    */
  final def unary_! () : Bool = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_! (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : Bool = this === 0.U(1.W)

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this << castToInt(that, "Shift amount")
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this >> castToInt(that, "Shift amount")
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), DynamicShiftRightOp, that)

  /** Conditionally set or clear a bit
    *
    * @param off a dynamic offset
    * @param dat set if true, clear if false
    * @return a hrdware $coll with bit `off` set or cleared based on the value of `dat`
    * $unchangedWidth
    */
  final def bitSet(off: UInt, dat: Bool): UInt = macro UIntTransform.bitset

  /** @group SourceInfoTransformMacro */
  def do_bitSet(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
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
  final def zext(): SInt = macro SourceInfoTransform.noArg
  /** @group SourceInfoTransformMacro */
  def do_zext(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width + 1), ConvertOp, ref))

  override def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width), AsSIntOp, ref))
  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this
  override def do_asFixedPoint(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
    }
  }

  override def do_asInterval(range: IntervalRange = IntervalRange.Unknown)
                            (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    (range.lower, range.upper, range.binaryPoint) match {
      case (lx: firrtlconstraint.IsKnown, ux: firrtlconstraint.IsKnown, KnownBinaryPoint(bp)) =>
        // No mechanism to pass open/close to firrtl so need to handle directly
        val l = lx match {
          case firrtlir.Open(x) => x + BigDecimal(1) / BigDecimal(BigInt(1) << bp)
          case firrtlir.Closed(x) => x
        }
        val u = ux match {
          case firrtlir.Open(x) => x - BigDecimal(1) / BigDecimal(BigInt(1) << bp)
          case firrtlir.Closed(x) => x
        }
        val minBI = (l * BigDecimal(BigInt(1) << bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        val maxBI = (u * BigDecimal(BigInt(1) << bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        pushOp(DefPrim(sourceInfo, Interval(range), AsIntervalOp, ref, ILit(minBI), ILit(maxBI), ILit(bp)))
      case _ =>
        throwException(
          s"cannot call $this.asInterval($range), you must specify a known binaryPoint and range")
    }
  }
  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that.asUInt
  }

  private def subtractAsSInt(that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that)
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
    val bindingString = litOption match {
      case Some(value) => s"($value)"
      case _ => bindingToString
    }
    s"SInt$width$bindingString"
  }

  private[chisel3] override def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass && this.width == that.width  // TODO: should this be true for unspecified widths?

  private[chisel3] override def cloneTypeWidth(w: Width): this.type =
    new SInt(w).asInstanceOf[this.type]

  /** Unary negation (expanding width)
    *
    * @return a hardware $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_- (): SInt = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus `this` shifted right by one
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_-% (): SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = 0.S - this
  /** @group SourceInfoTransformMacro */
  def unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = 0.S -% this

  /** add (default - no growth) operator */
  override def do_+ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    this +% that
  /** subtract (default - no growth) operator */
  override def do_- (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    this -% that
  override def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width + that.width), TimesOp, that)
  override def do_/ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width + 1), DivideOp, that)
  override def do_% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width min that.width), RemOp, that)

  /** Multiplication operator
    *
    * @param that a hardware $coll
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def * (that: UInt): SInt = macro SourceInfoTransform.thatArg
  /** @group SourceInfoTransformMacro */
  def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = {
    val thatToSInt = that.zext()
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
  final def +& (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  final def +% (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def -& (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  final def -% (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), AddOp, that)
  /** @group SourceInfoTransformMacro */
  def do_+% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    (this +& that).tail(1).asSInt
  /** @group SourceInfoTransformMacro */
  def do_-& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that)
  /** @group SourceInfoTransformMacro */
  def do_-% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    (this -& that).tail(1).asSInt

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def & (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def | (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def ^ (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that).asSInt
  /** @group SourceInfoTransformMacro */
  def do_| (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that).asSInt
  /** @group SourceInfoTransformMacro */
  def do_^ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that).asSInt

  /** @group SourceInfoTransformMacro */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    unop(sourceInfo, UInt(width = width), BitNotOp).asSInt

  override def do_< (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  final def =/= (that: SInt): Bool = macro SourceInfoTransform.thatArg

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  final def === (that: SInt): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_=/= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  /** @group SourceInfoTransformMacro */
  def do_=== (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)

//  final def abs(): UInt = macro SourceInfoTransform.noArg

  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = {
    Mux(this < 0.S, (-this), this)
  }

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    this << castToInt(that, "Shift amount")
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    this >> castToInt(that, "Shift amount")
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width), DynamicShiftRightOp, that)

  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
  override def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = this
  override def do_asFixedPoint(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
    }
  }

  override def do_asInterval(range: IntervalRange = IntervalRange.Unknown)
                            (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    (range.lower, range.upper, range.binaryPoint) match {
      case (lx: firrtlconstraint.IsKnown, ux: firrtlconstraint.IsKnown, KnownBinaryPoint(bp)) =>
        // No mechanism to pass open/close to firrtl so need to handle directly
        val l = lx match {
          case firrtlir.Open(x) => x + BigDecimal(1) / BigDecimal(BigInt(1) << bp)
          case firrtlir.Closed(x) => x
        }
        val u = ux match {
          case firrtlir.Open(x) => x - BigDecimal(1) / BigDecimal(BigInt(1) << bp)
          case firrtlir.Closed(x) => x
        }
        //TODO: (chick) Need to determine, what asInterval needs, and why it might need min and max as args -- CAN IT BE UNKNOWN?
        // Angie's operation: Decimal -> Int -> Decimal loses information. Need to be conservative here?
        val minBI = (l * BigDecimal(BigInt(1) << bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        val maxBI = (u * BigDecimal(BigInt(1) << bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        pushOp(DefPrim(sourceInfo, Interval(range), AsIntervalOp, ref, ILit(minBI), ILit(maxBI), ILit(bp)))
      case _ =>
        throwException(
          s"cannot call $this.asInterval($range), you must specify a known binaryPoint and range")
    }
  }

  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
    this := that.asSInt
  }
}

sealed trait Reset extends Element with ToBoolable {
  /** Casts this $coll to an [[AsyncReset]] */
  final def asAsyncReset(): AsyncReset = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): AsyncReset
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
  override def toString: String = s"Reset$bindingToString"

  def cloneType: this.type = Reset().asInstanceOf[this.type]

  private[chisel3] def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass

  override def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: Reset | DontCare => super.connect(that)(sourceInfo, connectCompileOptions)
    case _ => super.badConnect(that)(sourceInfo)
  }

  override def litOption = None

  /** Not really supported */
  def toPrintable: Printable = PString("Reset")

  override def do_asUInt(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))

  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that
  }

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): AsyncReset =
    pushOp(DefPrim(sourceInfo, AsyncReset(), AsAsyncResetOp, ref))

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), AsUIntOp, ref))

  /** @group SourceInfoTransformMacro */
  def do_toBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = do_asBool
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
  override def toString: String = s"AsyncReset$bindingToString"

  def cloneType: this.type = AsyncReset().asInstanceOf[this.type]

  private[chisel3] def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass

  override def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: AsyncReset | DontCare => super.connect(that)(sourceInfo, connectCompileOptions)
    case _ => super.badConnect(that)(sourceInfo)
  }

  override def litOption = None

  /** Not really supported */
  def toPrintable: Printable = PString("AsyncReset")

  override def do_asUInt(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))

  // TODO Is this right?
  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that.asBool.asAsyncReset
  }

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): AsyncReset = this

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), AsUIntOp, ref))

  /** @group SourceInfoTransformMacro */
  def do_toBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = do_asBool
}

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  *
  * @define coll [[Bool]]
  * @define numType $coll
  */
sealed class Bool() extends UInt(1.W) with Reset {
  override def toString: String = {
    val bindingString = litToBooleanOption match {
      case Some(value) => s"($value)"
      case _ => bindingToString
    }
    s"Bool$bindingString"
  }

  private[chisel3] override def cloneTypeWidth(w: Width): this.type = {
    require(!w.known || w.get == 1)
    new Bool().asInstanceOf[this.type]
  }

  /** Convert to a [[scala.Option]] of [[scala.Boolean]] */
  def litToBooleanOption: Option[Boolean] = litOption.map {
    case intVal if intVal == 1 => true
    case intVal if intVal == 0 => false
    case intVal => throwException(s"Boolean with unexpected literal value $intVal")
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
  final def & (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * @group Bitwise
    */
  final def | (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * @group Bitwise
    */
  final def ^ (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    binop(sourceInfo, Bool(), BitAndOp, that)
  /** @group SourceInfoTransformMacro */
  def do_| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    binop(sourceInfo, Bool(), BitOrOp, that)
  /** @group SourceInfoTransformMacro */
  def do_^ (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    binop(sourceInfo, Bool(), BitXorOp, that)

  /** @group SourceInfoTransformMacro */
  override def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    unop(sourceInfo, Bool(), BitNotOp)

  /** Logical or operator
    *
    * @param that a hardware $coll
    * @return the lgocial or of this $coll and `that`
    * @note this is equivalent to [[Bool!.|(that:chisel3\.Bool)* Bool.|)]]
    * @group Logical
    */
  def || (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_|| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this | that

  /** Logical and operator
    *
    * @param that a hardware $coll
    * @return the lgocial and of this $coll and `that`
    * @note this is equivalent to [[Bool!.&(that:chisel3\.Bool)* Bool.&]]
    * @group Logical
    */
  def && (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this & that

  /** Reinterprets this $coll as a clock */
  def asClock(): Clock = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asClock(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Clock = pushOp(DefPrim(sourceInfo, Clock(), AsClockOp, ref))

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): AsyncReset =
    pushOp(DefPrim(sourceInfo, AsyncReset(), AsAsyncResetOp, ref))
}

package experimental {

  import chisel3.internal.firrtl.BinaryPoint
  import chisel3.internal.requireIsHardware // Fix ambiguous import

  /** Chisel types that have binary points support retrieving
    * literal values as `Double` or `BigDecimal`
    */
  trait HasBinaryPoint { self: Bits =>
    def binaryPoint: BinaryPoint

    /** Return the [[Double]] value of this instance if it is a Literal
      * @note this method may throw an exception if the literal value won't fit in a Double
      */
    def litToDoubleOption: Option[Double] = {
      litOption match {
        case Some(bigInt: BigInt) =>
          Some(Num.toDouble(bigInt, binaryPoint))
        case _ => None
      }
    }

    /** Return the double value of this instance assuming it is a literal (convenience method)
      */
    def litToDouble: Double = litToDoubleOption.get

    /** Return the [[BigDecimal]] value of this instance if it is a Literal
      * @note this method may throw an exception if the literal value won't fit in a BigDecimal
      */
    def litToBigDecimalOption: Option[BigDecimal] = {
      litOption match {
        case Some(bigInt: BigInt) =>
          Some(Num.toBigDecimal(bigInt, binaryPoint))
        case _ => None
      }
    }

    /** Return the [[BigDecimal]] value of this instance assuming it is a literal (convenience method)
      * @return
      */
    def litToBigDecimal: BigDecimal = litToBigDecimalOption.get
  }
  /** A sealed class representing a fixed point number that has a bit width and a binary point The width and binary point
    * may be inferred.
    *
    * IMPORTANT: The API provided here is experimental and may change in the future.
    *
    * @param width       bit width of the fixed point number
    * @param binaryPoint the position of the binary point with respect to the right most bit of the width currently this
    *                    should be positive but it is hoped to soon support negative points and thus use this field as a
    *                    simple exponent
    * @define coll           [[FixedPoint]]
    * @define numType        $coll
    * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
    * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
    */
  sealed class FixedPoint private(width: Width, val binaryPoint: BinaryPoint)
    extends Bits(width) with Num[FixedPoint] with HasBinaryPoint {

    override def toString: String = {
      val bindingString = litToDoubleOption match {
        case Some(value) => s"($value)"
        case _ => bindingToString
      }
      s"FixedPoint$width$binaryPoint$bindingString"
    }

    private[chisel3] override def typeEquivalent(that: Data): Boolean = that match {
      case that: FixedPoint => this.width == that.width && this.binaryPoint == that.binaryPoint // TODO: should this be true for unspecified widths?
      case _ => false
    }

    private[chisel3] override def cloneTypeWidth(w: Width): this.type =
      new FixedPoint(w, binaryPoint).asInstanceOf[this.type]

    override def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
      case _: FixedPoint|DontCare => super.connect(that)
      case _ => this badConnect that
    }

    /** Unary negation (expanding width)
      *
      * @return a hardware $coll equal to zero minus this $coll
      *         $expandingWidth
      * @group Arithmetic
      */
    final def unary_- (): FixedPoint = macro SourceInfoTransform.noArg

    /** Unary negation (constant width)
      *
      * @return a hardware $coll equal to zero minus `this` shifted right by one
      *         $constantWidth
      * @group Arithmetic
      */
    final def unary_-% (): FixedPoint = macro SourceInfoTransform.noArg

    /** @group SourceInfoTransformMacro */
    def unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = FixedPoint.fromBigInt(0) - this
    /** @group SourceInfoTransformMacro */
    def unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = FixedPoint.fromBigInt(0) -% this

    /** add (default - no growth) operator */
    override def do_+ (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      this +% that
    /** subtract (default - no growth) operator */
    override def do_- (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      this -% that
    override def do_* (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      binop(sourceInfo, FixedPoint(this.width + that.width, this.binaryPoint + that.binaryPoint), TimesOp, that)
    override def do_/ (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      throwException(s"division is illegal on FixedPoint types")
    override def do_% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      throwException(s"mod is illegal on FixedPoint types")


    /** Multiplication operator
      *
      * @param that a hardware [[UInt]]
      * @return the product of this $coll and `that`
      *         $sumWidth
      *         $singleCycleMul
      * @group Arithmetic
      */
    final def * (that: UInt): FixedPoint = macro SourceInfoTransform.thatArg
    /** @group SourceInfoTransformMacro */
    def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      binop(sourceInfo, FixedPoint(this.width + that.width, binaryPoint), TimesOp, that)

    /** Multiplication operator
      *
      * @param that a hardware [[SInt]]
      * @return the product of this $coll and `that`
      *         $sumWidth
      *         $singleCycleMul
      * @group Arithmetic
      */
    final def * (that: SInt): FixedPoint = macro SourceInfoTransform.thatArg
    /** @group SourceInfoTransformMacro */
    def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      binop(sourceInfo, FixedPoint(this.width + that.width, binaryPoint), TimesOp, that)

    /** Addition operator (expanding width)
      *
      * @param that a hardware $coll
      * @return the sum of this $coll and `that`
      *         $maxWidthPlusOne
      * @group Arithmetic
      */
    final def +& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

    /** Addition operator (constant width)
      *
      * @param that a hardware $coll
      * @return the sum of this $coll and `that` shifted right by one
      *         $maxWidth
      * @group Arithmetic
      */
    final def +% (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

    /** Subtraction operator (increasing width)
      *
      * @param that a hardware $coll
      * @return the difference of this $coll less `that`
      *         $maxWidthPlusOne
      * @group Arithmetic
      */
    final def -& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

    /** Subtraction operator (constant width)
      *
      * @param that a hardware $coll
      * @return the difference of this $coll less `that` shifted right by one
      *         $maxWidth
      * @group Arithmetic
      */
    final def -% (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_+& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
      (this.width, that.width, this.binaryPoint, that.binaryPoint) match {
        case (KnownWidth(thisWidth), KnownWidth(thatWidth), KnownBinaryPoint(thisBP), KnownBinaryPoint(thatBP)) =>
          val thisIntWidth = thisWidth - thisBP
          val thatIntWidth = thatWidth - thatBP
          val newBinaryPoint = thisBP max thatBP
          val newWidth = (thisIntWidth max thatIntWidth) + newBinaryPoint + 1
          binop(sourceInfo, FixedPoint(newWidth.W, newBinaryPoint.BP), AddOp, that)
        case _ =>
          val newBinaryPoint = this.binaryPoint max that.binaryPoint
          binop(sourceInfo, FixedPoint(UnknownWidth(), newBinaryPoint), AddOp, that)
      }
    }

    /** @group SourceInfoTransformMacro */
    def do_+% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      (this +& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)
    /** @group SourceInfoTransformMacro */
    def do_-& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
      (this.width, that.width, this.binaryPoint, that.binaryPoint) match {
        case (KnownWidth(thisWidth), KnownWidth(thatWidth), KnownBinaryPoint(thisBP), KnownBinaryPoint(thatBP)) =>
          val thisIntWidth = thisWidth - thisBP
          val thatIntWidth = thatWidth - thatBP
          val newBinaryPoint = thisBP max thatBP
          val newWidth = (thisIntWidth max thatIntWidth) + newBinaryPoint + 1
          binop(sourceInfo, FixedPoint(newWidth.W, newBinaryPoint.BP), SubOp, that)
        case _ =>
          val newBinaryPoint = this.binaryPoint max that.binaryPoint
          binop(sourceInfo, FixedPoint(UnknownWidth(), newBinaryPoint), SubOp, that)
      }
    }

    /** @group SourceInfoTransformMacro */
    def do_-% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      (this -& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)

    /** Bitwise and operator
      *
      * @param that a hardware $coll
      * @return the bitwise and of  this $coll and `that`
      *         $maxWidth
      * @group Bitwise
      */
    final def & (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

    /** Bitwise or operator
      *
      * @param that a hardware $coll
      * @return the bitwise or of this $coll and `that`
      *         $maxWidth
      * @group Bitwise
      */
    final def | (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

    /** Bitwise exclusive or (xor) operator
      *
      * @param that a hardware $coll
      * @return the bitwise xor of this $coll and `that`
      *         $maxWidth
      * @group Bitwise
      */
    final def ^ (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      throwException(s"And is illegal between $this and $that")
    /** @group SourceInfoTransformMacro */
    def do_| (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      throwException(s"Or is illegal between $this and $that")
    /** @group SourceInfoTransformMacro */
    def do_^ (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      throwException(s"Xor is illegal between $this and $that")

    final def setBinaryPoint(that: Int): FixedPoint = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_setBinaryPoint(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = this.binaryPoint match {
      case KnownBinaryPoint(value) =>
        binop(sourceInfo, FixedPoint(this.width + (that - value), KnownBinaryPoint(that)), SetBinaryPoint, that)
      case _ =>
        binop(sourceInfo, FixedPoint(UnknownWidth(), KnownBinaryPoint(that)), SetBinaryPoint, that)
    }

    /** @group SourceInfoTransformMacro */
    def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      throwException(s"Not is illegal on $this")

    override def do_< (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
    override def do_> (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
    override def do_<= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
    override def do_>= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

    final def != (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg

    /** Dynamic not equals operator
      *
      * @param that a hardware $coll
      * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
      * @group Comparison
      */
    final def =/= (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg

    /** Dynamic equals operator
      *
      * @param that a hardware $coll
      * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
      * @group Comparison
      */
    final def === (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_!= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
    /** @group SourceInfoTransformMacro */
    def do_=/= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
    /** @group SourceInfoTransformMacro */
    def do_=== (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)

    def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
      // TODO: remove this once we have CompileOptions threaded through the macro system.
      import chisel3.ExplicitCompileOptions.NotStrict
      Mux(this < 0.F(0.BP), 0.F(0.BP) - this, this)
    }

    override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      binop(sourceInfo, FixedPoint(this.width + that, this.binaryPoint), ShiftLeftOp, validateShiftAmount(that))
    override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      (this << castToInt(that, "Shift amount")).asFixedPoint(this.binaryPoint)
    override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      binop(sourceInfo, FixedPoint(this.width.dynamicShiftLeft(that.width), this.binaryPoint), DynamicShiftLeftOp, that)
    override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      binop(sourceInfo, FixedPoint(this.width.shiftRight(that), this.binaryPoint), ShiftRightOp, validateShiftAmount(that))
    override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      (this >> castToInt(that, "Shift amount")).asFixedPoint(this.binaryPoint)
    override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      binop(sourceInfo, FixedPoint(this.width, this.binaryPoint), DynamicShiftRightOp, that)

    override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
    override def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = pushOp(DefPrim(sourceInfo, SInt(this.width), AsSIntOp, ref))

    override def do_asFixedPoint(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
      binaryPoint match {
        case KnownBinaryPoint(value) =>
          val iLit = ILit(value)
          pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
        case _ =>
          throwException(s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
      }
    }

  def do_asInterval(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    throwException(s"cannot call $this.asInterval(binaryPoint=$binaryPoint), you must specify a range")
  }

  override def do_asInterval(range: IntervalRange = IntervalRange.Unknown)
                            (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    (range.lower, range.upper, range.binaryPoint) match {
      case (lx: firrtlconstraint.IsKnown, ux: firrtlconstraint.IsKnown, KnownBinaryPoint(bp)) =>
        // No mechanism to pass open/close to firrtl so need to handle directly
        val l = lx match {
          case firrtlir.Open(x) => x + BigDecimal(1) / BigDecimal(BigInt(1) << bp)
          case firrtlir.Closed(x) => x
        }
        val u = ux match {
          case firrtlir.Open(x) => x - BigDecimal(1) / BigDecimal(BigInt(1) << bp)
          case firrtlir.Closed(x) => x
        }
        val minBI = (l * BigDecimal(BigInt(1) << bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        val maxBI = (u * BigDecimal(BigInt(1) << bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        pushOp(DefPrim(sourceInfo, Interval(range), AsIntervalOp, ref, ILit(minBI), ILit(maxBI), ILit(bp)))
      case _ =>
        throwException(
          s"cannot call $this.asInterval($range), you must specify a known binaryPoint and range")
    }
  }

    private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
      // TODO: redefine as just asFixedPoint on that, where FixedPoint.asFixedPoint just works.
      this := (that match {
        case fp: FixedPoint => fp.asSInt.asFixedPoint(this.binaryPoint)
        case _ => that.asFixedPoint(this.binaryPoint)
      })
    }
  }

  /** Use PrivateObject to force users to specify width and binaryPoint by name
    */
  sealed trait PrivateType
  private case object PrivateObject extends PrivateType

  /**
    * Factory and convenience methods for the FixedPoint class
    * IMPORTANT: The API provided here is experimental and may change in the future.
    */
  object FixedPoint extends NumObject {

    import FixedPoint.Implicits._

    /** Create an FixedPoint type with inferred width. */
    def apply(): FixedPoint = apply(Width(), BinaryPoint())

    /** Create an FixedPoint type or port with fixed width. */
    def apply(width: Width, binaryPoint: BinaryPoint): FixedPoint = new FixedPoint(width, binaryPoint)

    /** Create an FixedPoint literal with inferred width from BigInt.
      * Use PrivateObject to force users to specify width and binaryPoint by name
      */
    def fromBigInt(value: BigInt, width: Width, binaryPoint: BinaryPoint): FixedPoint = {
      apply(value, width, binaryPoint)
    }
    /** Create an FixedPoint literal with inferred width from BigInt.
      * Use PrivateObject to force users to specify width and binaryPoint by name
      */
    def fromBigInt(value: BigInt, binaryPoint: BinaryPoint = 0.BP): FixedPoint = {
      apply(value, Width(), binaryPoint)
    }
    /** Create an FixedPoint literal with inferred width from BigInt.
      * Use PrivateObject to force users to specify width and binaryPoint by name
      */
    def fromBigInt(value: BigInt, width: Int, binaryPoint: Int): FixedPoint =
      if(width == -1) {
        apply(value, Width(), BinaryPoint(binaryPoint))
      }
      else {
        apply(value, Width(width), BinaryPoint(binaryPoint))
      }
    /** Create an FixedPoint literal with inferred width from Double.
      * Use PrivateObject to force users to specify width and binaryPoint by name
      */
    def fromDouble(value: Double, width: Width, binaryPoint: BinaryPoint): FixedPoint = {
      fromBigInt(
        toBigInt(value, binaryPoint.get), width = width, binaryPoint = binaryPoint
      )
    }
    /** Create an FixedPoint literal with inferred width from BigDecimal.
      * Use PrivateObject to force users to specify width and binaryPoint by name
      */
    def fromBigDecimal(value: BigDecimal, width: Width, binaryPoint: BinaryPoint): FixedPoint = {
      fromBigInt(
        toBigInt(value, binaryPoint.get), width = width, binaryPoint = binaryPoint
      )
    }

    /** Create an FixedPoint port with specified width and binary position. */
    def apply(value: BigInt, width: Width, binaryPoint: BinaryPoint): FixedPoint = {
      val lit = FPLit(value, width, binaryPoint)
      val newLiteral = new FixedPoint(lit.width, lit.binaryPoint)
      // Ensure we have something capable of generating a name.
      lit.bindLitArg(newLiteral)
    }



    object Implicits {

      implicit class fromDoubleToLiteral(double: Double) {
        def F(binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, Width(), binaryPoint)
        }

        def F(width: Width, binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, width, binaryPoint)
        }
      }

      implicit class fromBigDecimalToLiteral(bigDecimal: BigDecimal) {
        def F(binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromBigDecimal(bigDecimal, Width(), binaryPoint)
        }

        def F(width: Width, binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromBigDecimal(bigDecimal, width, binaryPoint)
        }
      }
    }
  }

  /**
    * A sealed class representing a fixed point number that has a range, an additional
    * parameter that can determine a minimum and maximum supported value.
    * The range can be used to reduce the required widths particularly in primitive
    * operations with other Intervals, the canonical example being
    * {{{
    *   val one = 1.I
    *   val six = Seq.fill(6)(one).reduce(_ + _)
    * }}}
    * A UInt computed in this way would require a [[Width]]
    * binary point
    * The width and binary point may be inferred.
    *
    * IMPORTANT: The API provided here is experimental and may change in the future.
    *
    * @param range       a range specifies min, max and binary point
    */
  sealed class Interval private[chisel3] (val range: chisel3.internal.firrtl.IntervalRange)
    extends Bits(range.getWidth) with Num[Interval] with HasBinaryPoint {

    override def toString: String = {
      val bindingString = litOption match {
        case Some(value) => s"($value)"
        case _ => bindingToString
      }
      s"Interval$width$bindingString"
    }

    private[chisel3] override def cloneTypeWidth(w: Width): this.type =
      new Interval(range).asInstanceOf[this.type]

    def toType: String = {
      val zdec1 = """([+\-]?[0-9]\d*)(\.[0-9]*[1-9])(0*)""".r
      val zdec2 = """([+\-]?[0-9]\d*)(\.0*)""".r
      val dec = """([+\-]?[0-9]\d*)(\.[0-9]\d*)""".r
      val int = """([+\-]?[0-9]\d*)""".r
      def dec2string(v: BigDecimal): String = v.toString match {
        case zdec1(x, y, z) => x + y
        case zdec2(x, y) => x
        case other => other
      }

      val lowerString = range.lower match {
        case firrtlir.Open(l)      => s"(${dec2string(l)}, "
        case firrtlir.Closed(l)    => s"[${dec2string(l)}, "
        case firrtlir.UnknownBound => s"[?, "
        case _  => s"[?, "
      }
      val upperString = range.upper match {
        case firrtlir.Open(u)      => s"${dec2string(u)})"
        case firrtlir.Closed(u)    => s"${dec2string(u)}]"
        case firrtlir.UnknownBound => s"?]"
        case _  => s"?]"
      }
      val bounds = lowerString + upperString

      val pointString = range.binaryPoint match {
        case KnownBinaryPoint(i)  => "." + i.toString
        case _ => ""
      }
      "Interval" + bounds + pointString
    }

    private[chisel3] override def typeEquivalent(that: Data): Boolean =
      that.isInstanceOf[Interval] && this.width == that.width

    def binaryPoint: BinaryPoint = range.binaryPoint

    override def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = {
      that match {
        case _: Interval|DontCare => super.connect(that)
        case _ => this badConnect that
      }
    }

    final def unary_-(): Interval = macro SourceInfoTransform.noArg
    final def unary_-%(): Interval = macro SourceInfoTransform.noArg

    def unary_-(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      Interval.Zero - this
    }
    def unary_-%(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      Interval.Zero -% this
    }

    /** add (default - growing) operator */
    override def do_+(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      this +& that
    /** subtract (default - growing) operator */
    override def do_-(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      this -& that
    override def do_*(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      binop(sourceInfo, Interval(this.range * that.range), TimesOp, that)

    override def do_/(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      throwException(s"division is illegal on Interval types")
    override def do_%(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      throwException(s"mod is illegal on Interval types")

    /** add (width +1) operator */
    final def +&(that: Interval): Interval = macro SourceInfoTransform.thatArg
    /** add (no growth) operator */
    final def +%(that: Interval): Interval = macro SourceInfoTransform.thatArg
    /** subtract (width +1) operator */
    final def -&(that: Interval): Interval = macro SourceInfoTransform.thatArg
    /** subtract (no growth) operator */
    final def -%(that: Interval): Interval = macro SourceInfoTransform.thatArg

    def do_+&(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      binop(sourceInfo, Interval(this.range +& that.range), AddOp, that)
    }

    def do_+%(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      throwException(s"Non-growing addition is not supported on Intervals: ${sourceInfo}")
    }

    def do_-&(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      binop(sourceInfo, Interval(this.range -& that.range), SubOp, that)
    }

    def do_-%(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      throwException(s"Non-growing subtraction is not supported on Intervals: ${sourceInfo}, try squeeze")
    }

    final def &(that: Interval): Interval = macro SourceInfoTransform.thatArg
    final def |(that: Interval): Interval = macro SourceInfoTransform.thatArg
    final def ^(that: Interval): Interval = macro SourceInfoTransform.thatArg

    def do_&(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      throwException(s"And is illegal between $this and $that")
    def do_|(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      throwException(s"Or is illegal between $this and $that")
    def do_^(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      throwException(s"Xor is illegal between $this and $that")

    final def setPrecision(that: Int): Interval = macro SourceInfoTransform.thatArg

    // Precision change changes range -- see firrtl PrimOps (requires floor)
    // aaa.bbb -> aaa.bb for sbp(2)
    def do_setPrecision(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      val newBinaryPoint = BinaryPoint(that)
      val newIntervalRange = this.range.setPrecision(newBinaryPoint)
      binop(sourceInfo, Interval(newIntervalRange), SetBinaryPoint, that)
    }

    /** Increase the precision of this Interval, moves the binary point to the left.
      * aaa.bbb -> aaa.bbb00
      * @param that    how many bits to shift binary point
      * @return
      */
    final def increasePrecision(that: Int): Interval = macro SourceInfoTransform.thatArg

    def do_increasePrecision(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      assert(that > 0, s"Must increase precision by an integer greater than zero.")
      val newBinaryPoint = BinaryPoint(that)
      val newIntervalRange = this.range.incPrecision(newBinaryPoint)
      binop(sourceInfo, Interval(newIntervalRange), IncreasePrecision, that)
    }

    /** Decrease the precision of this Interval, moves the binary point to the right.
      * aaa.bbb -> aaa.b
      *
      * @param that    number of bits to move binary point
      * @return
      */
    final def decreasePrecision(that: Int): Interval = macro SourceInfoTransform.thatArg

    def do_decreasePrecision(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      assert(that > 0, s"Must decrease precision by an integer greater than zero.")
      val newBinaryPoint = BinaryPoint(that)
      val newIntervalRange = this.range.decPrecision(newBinaryPoint)
      binop(sourceInfo, Interval(newIntervalRange), DecreasePrecision, that)
    }

    /** Returns this wire bitwise-inverted. */
    def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      throwException(s"Not is illegal on $this")

    override def do_< (that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
    override def do_> (that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
    override def do_<= (that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
    override def do_>= (that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

    final def != (that: Interval): Bool = macro SourceInfoTransform.thatArg
    final def =/= (that: Interval): Bool = macro SourceInfoTransform.thatArg
    final def === (that: Interval): Bool = macro SourceInfoTransform.thatArg

    def do_!= (that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
    def do_=/= (that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
    def do_=== (that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)

    //  final def abs(): UInt = macro SourceInfoTransform.noArg

    def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      Mux(this < Interval.Zero, (Interval.Zero - this), this)
    }

    override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      binop(sourceInfo, Interval(this.range << that), ShiftLeftOp, that)

    override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      do_<<(that.toInt)

    override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      binop(sourceInfo, Interval(this.range << that), DynamicShiftLeftOp, that)
    }

    override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      binop(sourceInfo, Interval(this.range >> that), ShiftRightOp, that)
    }

    override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
      do_>>(that.toInt)

    override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      binop(sourceInfo, Interval(this.range >> that), DynamicShiftRightOp, that)
    }

    /**
      * Squeeze returns the intersection of the ranges this interval and that Interval
      * Ignores binary point of argument
      * Treat as an unsafe cast; gives undefined behavior if this signal's value is outside of the resulting range
      * Adds no additional hardware; this strictly an unsafe type conversion to use at your own risk
      * @param that
      * @return
      */
    final def squeeze(that: Interval): Interval = macro SourceInfoTransform.thatArg
    def do_squeeze(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      val other = that
      requireIsHardware(this, s"'this' ($this)")
      requireIsHardware(other, s"'other' ($other)")
      pushOp(DefPrim(sourceInfo, Interval(this.range.squeeze(that.range)), SqueezeOp, this.ref, other.ref))
    }

    /**
      * Squeeze returns the intersection of the ranges this interval and that UInt
      * Currently, that must have a defined width
      * Treat as an unsafe cast; gives undefined behavior if this signal's value is outside of the resulting range
      * Adds no additional hardware; this strictly an unsafe type conversion to use at your own risk
      * @param that an UInt whose properties determine the squeezing
      * @return
      */
    final def squeeze(that: UInt): Interval = macro SourceInfoTransform.thatArg
    def do_squeeze(that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      that.widthOption match {
        case Some(w) =>
          do_squeeze(Wire(Interval(IntervalRange(that.width, BinaryPoint(0)))))
        case _ =>
          throwException(s"$this.squeeze($that) requires an UInt argument with a known width")
      }
    }

    /**
      * Squeeze returns the intersection of the ranges this interval and that SInt
      * Currently, that must have a defined width
      * Treat as an unsafe cast; gives undefined behavior if this signal's value is outside of the resulting range
      * Adds no additional hardware; this strictly an unsafe type conversion to use at your own risk
      * @param that an SInt whose properties determine the squeezing
      * @return
      */
    final def squeeze(that: SInt): Interval = macro SourceInfoTransform.thatArg
    def do_squeeze(that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      that.widthOption match {
        case Some(w) =>
          do_squeeze(Wire(Interval(IntervalRange(that.width, BinaryPoint(0)))))
        case _ =>
          throwException(s"$this.squeeze($that) requires an SInt argument with a known width")
      }
    }

    /**
      * Squeeze returns the intersection of the ranges this interval and that IntervalRange
      * Ignores binary point of argument
      * Treat as an unsafe cast; gives undefined behavior if this signal's value is outside of the resulting range
      * Adds no additional hardware; this strictly an unsafe type conversion to use at your own risk
      * @param that an Interval whose properties determine the squeezing
      * @return
      */
    final def squeeze(that: IntervalRange): Interval = macro SourceInfoTransform.thatArg
    def do_squeeze(that: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      val intervalLitOpt = Interval.getSmallestLegalLit(that)
      val intervalLit    = intervalLitOpt.getOrElse(
        throwException(s"$this.squeeze($that) requires an Interval range with known lower and upper bounds")
      )
      do_squeeze(intervalLit)
    }


    /**
      * Wrap the value of this [[Interval]] into the range of a different Interval with a presumably smaller range.
      * Ignores binary point of argument
      * Errors if requires wrapping more than once
      * @param that
      * @return
      */
    final def wrap(that: Interval): Interval = macro SourceInfoTransform.thatArg

    def do_wrap(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      val other = that
      requireIsHardware(this, s"'this' ($this)")
      requireIsHardware(other, s"'other' ($other)")
      pushOp(DefPrim(sourceInfo, Interval(this.range.wrap(that.range)), WrapOp, this.ref, other.ref))
    }

    /**
      * Wrap this interval into the range determined by that UInt
      * Errors if requires wrapping more than once
      * @param that an UInt whose properties determine the wrap
      * @return
      */
    final def wrap(that: UInt): Interval = macro SourceInfoTransform.thatArg
    def do_wrap(that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      that.widthOption match {
        case Some(w) =>
          val u = BigDecimal(BigInt(1) << w) - 1
          do_wrap(0.U.asInterval(IntervalRange(firrtlir.Closed(0), firrtlir.Closed(u), BinaryPoint(0))))
        case _ =>
          throwException(s"$this.wrap($that) requires UInt with known width")
      }
    }

    /**
      * Wrap this interval into the range determined by an SInt
      * Errors if requires wrapping more than once
      * @param that an SInt whose properties determine the bounds of the wrap
      * @return
      */
    final def wrap(that: SInt): Interval = macro SourceInfoTransform.thatArg
    def do_wrap(that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      that.widthOption match {
        case Some(w) =>
          val l = -BigDecimal(BigInt(1) << (that.getWidth - 1))
          val u = BigDecimal(BigInt(1) << (that.getWidth - 1)) - 1
          do_wrap(Wire(Interval(IntervalRange(firrtlir.Closed(l), firrtlir.Closed(u), BinaryPoint(0)))))
        case _ =>
          throwException(s"$this.wrap($that) requires SInt with known width")
      }
    }

    /**
      * Wrap this interval into the range determined by an IntervalRange
      * Adds hardware to change values outside of wrapped range to be at the boundary
      * Errors if requires wrapping more than once
      * Ignores binary point of argument
      * @param that an Interval whose properties determine the bounds of the wrap
      * @return
      */
    final def wrap(that: IntervalRange): Interval = macro SourceInfoTransform.thatArg
    def do_wrap(that: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      (that.lowerBound, that.upperBound) match {
        case (lower: firrtlconstraint.IsKnown, upperBound: firrtlconstraint.IsKnown) =>
          do_wrap(0.U.asInterval(IntervalRange(that.lowerBound, that.upperBound, BinaryPoint(0))))
        case _ =>
          throwException(s"$this.wrap($that) requires Interval argument with known lower and upper bounds")
      }
    }

    /**
      * Clip this interval into the range determined by argument's range
      * Adds hardware to change values outside of clipped range to be at the boundary
      * Ignores binary point of argument
      * @param that an Interval whose properties determine the clipping
      * @return
      */
    final def clip(that: Interval): Interval = macro SourceInfoTransform.thatArg
    def do_clip(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      binop(sourceInfo, Interval(this.range.clip(that.range)), ClipOp, that)
    }

    /**
      * Clip this interval into the range determined by argument's range
      * Adds hardware to change values outside of clipped range to be at the boundary
      * @param that an UInt whose width determines the clipping
      * @return
      */
    final def clip(that: UInt): Interval = macro SourceInfoTransform.thatArg
    def do_clip(that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      require(that.widthKnown, "UInt clip width must be known")
      val u = BigDecimal(BigInt(1) << that.getWidth) - 1
      do_clip(Wire(Interval(IntervalRange(firrtlir.Closed(0), firrtlir.Closed(u), BinaryPoint(0)))))
    }

    /**
      * Clip this interval into the range determined by argument's range
      * Adds hardware to move values outside of clipped range to the boundary
      * @param that   an SInt whose width determines the clipping
      * @return
      */
    final def clip(that: SInt): Interval = macro SourceInfoTransform.thatArg
    def do_clip(that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      require(that.widthKnown, "SInt clip width must be known")
      val l = -BigDecimal(BigInt(1) << (that.getWidth - 1))
      val u = BigDecimal(BigInt(1) << (that.getWidth - 1)) - 1
      do_clip(Wire(Interval(IntervalRange(firrtlir.Closed(l), firrtlir.Closed(u), BinaryPoint(0)))))
    }

    /**
      * Clip this interval into the range determined by argument's range
      * Adds hardware to move values outside of clipped range to the boundary
      * Ignores binary point of argument
      * @param that   an SInt whose width determines the clipping
      * @return
      */
    final def clip(that: IntervalRange): Interval = macro SourceInfoTransform.thatArg
    def do_clip(that: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      (that.lowerBound, that.upperBound) match {
        case (lower: firrtlconstraint.IsKnown, upperBound: firrtlconstraint.IsKnown) =>
          do_clip(0.U.asInterval(IntervalRange(that.lowerBound, that.upperBound, BinaryPoint(0))))
        case _ =>
          throwException(s"$this.clip($that) requires Interval argument with known lower and upper bounds")
      }
    }

    override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
      pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
    }
    override def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = {
      pushOp(DefPrim(sourceInfo, SInt(this.width), AsSIntOp, ref))
    }

    override def do_asFixedPoint(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
      binaryPoint match {
        case KnownBinaryPoint(value) =>
          val iLit = ILit(value)
          pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
        case _ =>
          throwException(
            s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
      }
    }

    // TODO: intervals chick INVALID -- not enough args
    def do_asInterval(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
      pushOp(DefPrim(sourceInfo, Interval(this.range), AsIntervalOp, ref))
      throwException(s"($this).asInterval must specify arguments INVALID")
    }

    // TODO:(chick) intervals chick looks like this is wrong and only for FP?
    def do_fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = {
      /*val res = Wire(this, null).asInstanceOf[this.type]
      res := (that match {
        case fp: FixedPoint => fp.asSInt.asFixedPoint(this.binaryPoint)
        case _ => that.asFixedPoint(this.binaryPoint)
      })
      res*/
      throwException("fromBits INVALID for intervals")
    }

    private[chisel3] override def connectFromBits(that: Bits)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
      this := that.asInterval(this.range)
    }
  }

  /** Use PrivateObject to force users to specify width and binaryPoint by name
    */

  /**
    * Factory and convenience methods for the Interval class
    * IMPORTANT: The API provided here is experimental and may change in the future.
    */
  object Interval extends NumObject {
    /** Create an Interval type with inferred width and binary point. */
    def apply(): Interval = Interval(range"[?,?]")

    /** Create an Interval type with specified width. */
    def apply(binaryPoint: BinaryPoint): Interval = {
      val binaryPointString = binaryPoint match {
        case KnownBinaryPoint(value) => s"$value"
        case _ => s""
      }
      Interval(range"[?,?].$binaryPointString")
    }

    /** Create an Interval type with specified width. */
    def apply(width: Width): Interval = Interval(width, 0.BP)

    /** Create an Interval type with specified width and binary point */
    def apply(width: Width, binaryPoint: BinaryPoint): Interval = {
      Interval(IntervalRange(width, binaryPoint))
    }

    /** Create an Interval type with specified range.
      * @param range  defines the properties
      */
    def apply(range: IntervalRange): Interval = {
      new Interval(range)
    }

    /** Creates a Interval connected to a Interval literal with the value zero */
    def Zero: Interval = Lit(0, 1.W, 0.BP)

    /** Creates an Interval zero that supports the given range
      * Useful for creating a Interval register that has a desired number of bits
      * {{{
      *   val myRegister = RegInit(Interval.Zero(r"[0,12]")
      * }}}
      * @param range
      * @return
      */
    def Zero(range: IntervalRange): Interval = Lit(0, range)

    /** Make an interval from this BigInt, the BigInt is treated as bits
      * So lower binaryPoint number of bits will treated as mantissa
      *
      * @param value
      * @param width
      * @param binaryPoint
      * @return
      */
    def fromBigInt(value: BigInt, width: Width = Width(), binaryPoint: BinaryPoint = 0.BP): Interval = {
      Interval.Lit(value, Width(), binaryPoint)
    }

    /** Create an Interval literal with inferred width from Double.
      * Use PrivateObject to force users to specify width and binaryPoint by name
      */
    def fromDouble(value: Double, dummy: PrivateType = PrivateObject,
                   width: Width, binaryPoint: BinaryPoint): Interval = {
      fromBigInt(
        toBigInt(value, binaryPoint), width = width, binaryPoint = binaryPoint
      )
    }

    /** Create an Interval literal with inferred width from Double.
      * Use PrivateObject to force users to specify width and binaryPoint by name
      */
    def fromBigDecimal(value: Double, dummy: PrivateType = PrivateObject,
                       width: Width, binaryPoint: BinaryPoint): Interval = {
      fromBigInt(
        toBigInt(value, binaryPoint), width = width, binaryPoint = binaryPoint
      )
    }

    protected[chisel3] def Lit(value: BigInt, width: Width, binaryPoint: BinaryPoint): Interval = {
      width match {
        case KnownWidth(w) =>
          if(value >= 0 && value.bitLength >= w || value < 0 && value.bitLength > w) {
            throw new ChiselException(
              s"Error literal interval value $value is too many bits for specified width $w"
            )
          }
        case _ =>
      }
      val lit = IntervalLit(value, width, binaryPoint)
      val bound = firrtlir.Closed(Interval.toBigDecimal(value, binaryPoint.asInstanceOf[KnownBinaryPoint].value))
      val result = new Interval(IntervalRange(bound, bound, binaryPoint))
      lit.bindLitArg(result)
    }

    protected[chisel3] def Lit(value: BigInt, range: IntervalRange): Interval = {
      val lit = IntervalLit(value, range.getWidth, range.binaryPoint)
      val bigDecimal = BigDecimal(value) / (1 << lit.binaryPoint.get)
      val inRange = (range.lowerBound, range.upperBound) match {
        case (firrtlir.Closed(l), firrtlir.Closed(u)) => l <= bigDecimal && bigDecimal <= u
        case (firrtlir.Closed(l), firrtlir.Open(u))   => l <= bigDecimal && bigDecimal < u
        case (firrtlir.Open(l), firrtlir.Closed(u))   => l < bigDecimal && bigDecimal <= u
        case (firrtlir.Open(l), firrtlir.Open(u))     => l < bigDecimal && bigDecimal < u
      }
      if(! inRange) {
        throw new ChiselException(
          s"Error literal interval value $bigDecimal is not contained in specified range $range"
        )
      }
      val result = Interval(range)
      lit.bindLitArg(result)
    }

    /**
      * This returns the smallest Interval literal that can legally fit in range, if possible
      * If the lower bound or binary point is not known then return None
      *
      * @param range use to figure low number
      * @return
      */
    def getSmallestLegalLit(range: IntervalRange): Option[Interval] = {
      val bp = range.binaryPoint
      range.lowerBound match {
        case firrtlir.Closed(lowerBound) =>
          Some(Interval.Lit(toBigInt(lowerBound.toDouble, bp), width = range.getWidth, bp))
        case firrtlir.Open(lowerBound) =>
          Some(Interval.Lit(toBigInt(lowerBound.toDouble, bp) + BigInt(1), width = range.getWidth, bp))
        case _ =>
          None
      }
    }

    /**
      * This returns the largest Interval literal that can legally fit in range, if possible
      * If the upper bound or binary point is not known then return None
      *
      * @param range use to figure low number
      * @return
      */
    def getLargestLegalLit(range: IntervalRange): Option[Interval] = {
      val bp = range.binaryPoint
      range.upperBound match {
        case firrtlir.Closed(upperBound) =>
          Some(Interval.Lit(toBigInt(upperBound.toDouble, bp), width = range.getWidth, bp))
        case firrtlir.Open(upperBound) =>
          Some(Interval.Lit(toBigInt(upperBound.toDouble, bp) - BigInt(1), width = range.getWidth, bp))
        case _ =>
          None
      }
    }

    /** Contains the implicit classes used to provide the .I methods to create intervals
      * from the standard numberic types.
      * {{{
      *   val x = 7.I
      *   val y = 7.5.I(4.BP)
      * }}}
      */
    object Implicits {
      implicit class fromBigIntToLiteralInterval(bigInt: BigInt) {
        def I: Interval = {
          Interval.Lit(bigInt, width = Width(), 0.BP)
        }

        def I(binaryPoint: BinaryPoint): Interval = {
          Interval.Lit(bigInt, width = Width(), binaryPoint = binaryPoint)
        }

        def I(width: Width, binaryPoint: BinaryPoint): Interval = {
          Interval.Lit(bigInt, width, binaryPoint)
        }

        def I(range: IntervalRange): Interval = {
          Interval.Lit(bigInt, range)
        }
      }

      implicit class fromIntToLiteralInterval(int: Int) extends fromBigIntToLiteralInterval(int)
      implicit class fromLongToLiteralInterval(long: Long) extends fromBigIntToLiteralInterval(long)

      implicit class fromBigDecimalToLiteralInterval(bigDecimal: BigDecimal) {
        def I: Interval = {
          Interval.Lit(Interval.toBigInt(bigDecimal, 0.BP), width = Width(), 0.BP)
        }

        def I(binaryPoint: BinaryPoint): Interval = {
          Interval.Lit(Interval.toBigInt(bigDecimal, binaryPoint), width = Width(), binaryPoint = binaryPoint)
        }

        def I(width: Width, binaryPoint: BinaryPoint): Interval = {
          Interval.Lit(Interval.toBigInt(bigDecimal, binaryPoint), width, binaryPoint)
        }

        def I(range: IntervalRange): Interval = {
          Interval.Lit(Interval.toBigInt(bigDecimal, range.binaryPoint), range)
        }
      }

      implicit class fromDoubleToLiteralInterval(double: Double)
        extends fromBigDecimalToLiteralInterval(BigDecimal(double))
    }
  }
}


