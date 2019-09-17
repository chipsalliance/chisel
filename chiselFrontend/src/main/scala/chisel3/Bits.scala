// See LICENSE for license details.

package chisel3

import scala.language.experimental.macros

import chisel3.experimental.FixedPoint
import chisel3.internal._
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform,
  UIntTransform}
import chisel3.internal.firrtl.PrimOp._

// scalastyle:off method.name line.size.limit file.size.limit

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

  /** Casts this $coll to a [[Bool]]
    *
    * @note The width must be known and equal to 1
    */
  final def toBool(): Bool = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_toBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool
}

/** A data type for values represented by a single bitvector. This provides basic bitwise operations.
  *
  * @groupdesc Bitwise Bitwise hardware operators
  * @define coll [[Bits]]
  * @define sumWidthInt    @note The width of the returned $coll is `width of this` + `that`.
  * @define sumWidth       @note The width of the returned $coll is `width of this` + `width of that`.
  * @define unchangedWidth @note The width of the returned $coll is unchanged, i.e., the `width of this`.
  */
sealed abstract class Bits(private[chisel3] val width: Width) extends Element with ToBoolable { //scalastyle:off number.of.methods
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

  /** @group SourceInfoTransformMacro */
  @chiselRuntimeDeprecated
  @deprecated("Use asBools instead", "3.2")
  def do_toBools(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Seq[Bool] = do_asBools

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

  final def do_asBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    width match {
      case KnownWidth(1) => this(0)
      case _ => throwException(s"can't covert ${this.getClass.getSimpleName}$width to Bool")
    }
  }

  @chiselRuntimeDeprecated
  @deprecated("Use asBool instead", "3.2")
  final def do_toBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = do_asBool

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
    binop(sourceInfo, UInt(this.width), RemOp, that)
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
  def do_orR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= 0.U
  /** @group SourceInfoTransformMacro */
  def do_andR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = width match {
    // Generate a simpler expression if the width is known
    case KnownWidth(w) => this === ((BigInt(1) << w) - 1).U
    case UnknownWidth() =>  ~this === 0.U
  }
  /** @group SourceInfoTransformMacro */
  def do_xorR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = redop(sourceInfo, XorReduceOp)

  override def do_< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  @chiselRuntimeDeprecated
  @deprecated("Use '=/=', which avoids potential precedence problems", "3.0")
  final def != (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= that

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
    binop(sourceInfo, SInt(this.width), DivideOp, that)
  override def do_% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width), RemOp, that)

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

  @chiselRuntimeDeprecated
  @deprecated("Use '=/=', which avoids potential precedence problems", "3.0")
  final def != (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= that

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

  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
    this := that.asSInt
  }
}

object SInt extends SIntFactory

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
    case _: Reset => super.connect(that)(sourceInfo, connectCompileOptions)
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
    case _: AsyncReset => super.connect(that)(sourceInfo, connectCompileOptions)
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

object Bool extends BoolFactory

package experimental {
  //scalastyle:off number.of.methods
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
    extends Bits(width) with Num[FixedPoint] {
    import FixedPoint.Implicits._

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

    /** Convert to a [[scala.Option]] of [[scala.Boolean]] */
    def litToDoubleOption: Option[Double] = litOption.map { intVal =>
      val multiplier = math.pow(2, binaryPoint.get)
      intVal.toDouble / multiplier
    }

    /** Convert to a [[scala.Option]] */
    def litToDouble: Double = litToDoubleOption.get


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

    // TODO(chick): Consider comparison with UInt and SInt
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

    private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
      // TODO: redefine as just asFixedPoint on that, where FixedPoint.asFixedPoint just works.
      this := (that match {
        case fp: FixedPoint => fp.asSInt.asFixedPoint(this.binaryPoint)
        case _ => that.asFixedPoint(this.binaryPoint)
      })
    }
    //TODO(chick): Consider "convert" as an arithmetic conversion to UInt/SInt
  }

  /** Use PrivateObject to force users to specify width and binaryPoint by name
    */
  sealed trait PrivateType
  private case object PrivateObject extends PrivateType

  /**
    * Factory and convenience methods for the FixedPoint class
    * IMPORTANT: The API provided here is experimental and may change in the future.
    */
  object FixedPoint {

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

    /** Create an FixedPoint port with specified width and binary position. */
    def apply(value: BigInt, width: Width, binaryPoint: BinaryPoint): FixedPoint = {
      val lit = FPLit(value, width, binaryPoint)
      val newLiteral = new FixedPoint(lit.width, lit.binaryPoint)
      // Ensure we have something capable of generating a name.
      lit.bindLitArg(newLiteral)
    }

    /**
      * How to create a bigint from a double with a specific binaryPoint
      * @param x           a double value
      * @param binaryPoint a binaryPoint that you would like to use
      * @return
      */
    def toBigInt(x: Double, binaryPoint: Int): BigInt = {
      val multiplier = math.pow(2, binaryPoint)
      val result = BigInt(math.round(x * multiplier))
      result
    }

    /**
      * converts a bigInt with the given binaryPoint into the double representation
      * @param i           a bigint
      * @param binaryPoint the implied binaryPoint of @i
      * @return
      */
    def toDouble(i: BigInt, binaryPoint: Int): Double = {
      val multiplier = math.pow(2, binaryPoint)
      val result = i.toDouble / multiplier
      result
    }

    object Implicits {

  //      implicit class fromDoubleToLiteral(val double: Double) extends AnyVal {
      implicit class fromDoubleToLiteral(double: Double) {
        def F(binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, Width(), binaryPoint)
        }

        def F(width: Width, binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, width, binaryPoint)
        }
      }

  //      implicit class fromIntToBinaryPoint(val int: Int) extends AnyVal {
      implicit class fromIntToBinaryPoint(int: Int) {
        def BP: BinaryPoint = BinaryPoint(int) // scalastyle:ignore method.name
      }

    }
  }
}
