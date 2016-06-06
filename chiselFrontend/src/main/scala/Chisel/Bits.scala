// See LICENSE for license details.

package Chisel

import scala.language.experimental.macros

import internal._
import internal.Builder.pushOp
import internal.firrtl._
import internal.sourceinfo.{SourceInfo, DeprecatedSourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform,
  UIntTransform, MuxTransform}
import firrtl.PrimOp._

/** Element is a leaf data type: it cannot contain other Data objects. Example
  * uses are for representing primitive data types, like integers and bits.
  */
abstract class Element(dirArg: Direction, val width: Width) extends Data(dirArg)

/** A data type for values represented by a single bitvector. Provides basic
  * bitwise operations.
  */
sealed abstract class Bits(dirArg: Direction, width: Width, override val litArg: Option[LitArg])
    extends Element(dirArg, width) {
  // TODO: perhaps make this concrete?
  // Arguments for: self-checking code (can't do arithmetic on bits)
  // Arguments against: generates down to a FIRRTL UInt anyways

  private[Chisel] def fromInt(x: BigInt, w: Int): this.type

  private[Chisel] def flatten: IndexedSeq[Bits] = IndexedSeq(this)

  def cloneType: this.type = cloneTypeWidth(width)

  override def <> (that: Data)(implicit sourceInfo: SourceInfo): Unit = this := that

  final def tail(n: Int): UInt = macro SourceInfoTransform.nArg
  final def head(n: Int): UInt = macro SourceInfoTransform.nArg

  def do_tail(n: Int)(implicit sourceInfo: SourceInfo): UInt = {
    val w = width match {
      case KnownWidth(x) =>
        require(x >= n, s"Can't tail($n) for width $x < $n")
        Width(x - n)
      case UnknownWidth() => Width()
    }
    binop(sourceInfo, UInt(width = w), TailOp, n)
  }


  def do_head(n: Int)(implicit sourceInfo: SourceInfo): UInt = {
    width match {
      case KnownWidth(x) => require(x >= n, s"Can't head($n) for width $x < $n")
      case UnknownWidth() =>
    }
    binop(sourceInfo, UInt(width = n), HeadOp, n)
  }

  /** Returns the specified bit on this wire as a [[Bool]], statically
    * addressed.
    */
  final def apply(x: BigInt): Bool = macro SourceInfoTransform.xArg

  final def do_apply(x: BigInt)(implicit sourceInfo: SourceInfo): Bool = {
    if (x < 0) {
      Builder.error(s"Negative bit indices are illegal (got $x)")
    }
    if (isLit()) {
      Bool(((litValue() >> x.toInt) & 1) == 1)
    } else {
      pushOp(DefPrim(sourceInfo, Bool(), BitsExtractOp, this.ref, ILit(x), ILit(x)))
    }
  }

  /** Returns the specified bit on this wire as a [[Bool]], statically
    * addressed.
    *
    * @note convenience method allowing direct use of Ints without implicits
    */
  final def apply(x: Int): Bool = macro SourceInfoTransform.xArg

  final def do_apply(x: Int)(implicit sourceInfo: SourceInfo): Bool = apply(BigInt(x))

  /** Returns the specified bit on this wire as a [[Bool]], dynamically
    * addressed.
    */
  final def apply(x: UInt): Bool = macro SourceInfoTransform.xArg

  final def do_apply(x: UInt)(implicit sourceInfo: SourceInfo): Bool = {
    (this >> x)(0)
  }

  /** Returns a subset of bits on this wire from `hi` to `lo` (inclusive),
    * statically addressed.
    *
    * @example
    * {{{
    * myBits = 0x5 = 0b101
    * myBits(1,0) => 0b01  // extracts the two least significant bits
    * }}}
    */
  final def apply(x: Int, y: Int): UInt = macro SourceInfoTransform.xyArg

  final def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo): UInt = {
    if (x < y || y < 0) {
      Builder.error(s"Invalid bit range ($x,$y)")
    }
    val w = x - y + 1
    if (isLit()) {
      UInt((litValue >> y) & ((BigInt(1) << w) - 1), w)
    } else {
      pushOp(DefPrim(sourceInfo, UInt(width = w), BitsExtractOp, this.ref, ILit(x), ILit(y)))
    }
  }

  // REVIEW TODO: again, is this necessary? Or just have this and use implicits?
  final def apply(x: BigInt, y: BigInt): UInt = macro SourceInfoTransform.xyArg

  final def do_apply(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    apply(x.toInt, y.toInt)

  private[Chisel] def unop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp): T =
    pushOp(DefPrim(sourceInfo, dest, op, this.ref))
  private[Chisel] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: BigInt): T =
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, ILit(other)))
  private[Chisel] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: Bits): T =
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, other.ref))

  private[Chisel] def compop(sourceInfo: SourceInfo, op: PrimOp, other: Bits): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  private[Chisel] def redop(sourceInfo: SourceInfo, op: PrimOp): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref))

  /** Returns this wire zero padded up to the specified width.
    *
    * @note for SInts only, this does sign extension
    */
  final def pad(that: Int): this.type = macro SourceInfoTransform.thatArg

  def do_pad(that: Int)(implicit sourceInfo: SourceInfo): this.type =
    binop(sourceInfo, cloneTypeWidth(this.width max Width(that)), PadOp, that)

  /** Returns this wire bitwise-inverted. */
  final def unary_~ (): Bits = macro SourceInfoWhiteboxTransform.noArg

  def do_unary_~ (implicit sourceInfo: SourceInfo): Bits


  /** Shift left operation */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or Bits?
  final def << (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo): Bits

  /** Returns this wire statically left shifted by the specified amount,
    * inserting zeros into the least significant bits.
    *
    * The width of the output is `other` larger than the input.
    */
  final def << (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_<< (that: Int)(implicit sourceInfo: SourceInfo): Bits

  /** Returns this wire dynamically left shifted by the specified amount,
    * inserting zeros into the least significant bits.
    *
    * The width of the output is `pow(2, width(other))` larger than the input.
    */
  final def << (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_<< (that: UInt)(implicit sourceInfo: SourceInfo): Bits

  /** Shift right operation */
  // REVIEW TODO: redundant
  final def >> (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo): Bits

  /** Returns this wire statically right shifted by the specified amount,
    * inserting zeros into the most significant bits.
    *
    * The width of the output is the same as the input.
    */
  final def >> (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_>> (that: Int)(implicit sourceInfo: SourceInfo): Bits

  /** Returns this wire dynamically right shifted by the specified amount,
    * inserting zeros into the most significant bits.
    *
    * The width of the output is the same as the input.
    */
  final def >> (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_>> (that: UInt)(implicit sourceInfo: SourceInfo): Bits

  /** Returns the contents of this wire as a [[Vec]] of [[Bool]]s.
    */
  final def toBools(): Seq[Bool] = macro SourceInfoTransform.noArg

  def toBools(implicit sourceInfo: SourceInfo): Seq[Bool] =
    Seq.tabulate(this.getWidth)(i => this(i))

  /** Reinterpret cast to a SInt.
    *
    * @note value not guaranteed to be preserved: for example, an UInt of width
    * 3 and value 7 (0b111) would become a SInt with value -1
    */
  final def asSInt(): SInt = macro SourceInfoTransform.noArg

  def do_asSInt(implicit sourceInfo: SourceInfo): SInt

  /** Reinterpret cast to Bits. */
  final def asBits(): Bits = macro SourceInfoTransform.noArg

  def do_asBits(implicit sourceInfo: SourceInfo): Bits = asUInt()

  @deprecated("Use asSInt, which makes the reinterpret cast more explicit", "chisel3")
  final def toSInt(): SInt = do_asSInt(DeprecatedSourceInfo)
  @deprecated("Use asUInt, which makes the reinterpret cast more explicit", "chisel3")
  final def toUInt(): UInt = do_asUInt(DeprecatedSourceInfo)

  final def toBool(): Bool = macro SourceInfoTransform.noArg

  def do_toBool(implicit sourceInfo: SourceInfo): Bool = {
    width match {
    case KnownWidth(1) => this(0)
    case _ => throwException(s"can't covert UInt<$width> to Bool")
  }
  }

  /** Returns this wire concatenated with `other`, where this wire forms the
    * most significant part and `other` forms the least significant part.
    *
    * The width of the output is sum of the inputs.
    */
  final def ## (that: Bits): UInt = macro SourceInfoTransform.thatArg

  def do_## (that: Bits)(implicit sourceInfo: SourceInfo): UInt = {
    val w = this.width + that.width
    pushOp(DefPrim(sourceInfo, UInt(w), ConcatOp, this.ref, that.ref))
  }

  override def do_fromBits(that: Bits)(implicit sourceInfo: SourceInfo): this.type = {
    val res = Wire(this, null).asInstanceOf[this.type]
    res := that
    res
  }
}

/** Provides a set of operations to create UInt types and literals.
  * Identical in functionality to the UInt companion object.
  */
object Bits extends UIntFactory

// REVIEW TODO: Further discussion needed on what Num actually is.
/** Abstract trait defining operations available on numeric-like wire data
  * types.
  */
abstract trait Num[T <: Data] {
  // def << (b: T): T
  // def >> (b: T): T
  //def unary_-(): T

  // REVIEW TODO: double check ops conventions against FIRRTL

  /** Outputs the sum of `this` and `b`. The resulting width is the max of the
    * operands plus 1 (should not overflow).
    */
  final def + (that: T): T = macro SourceInfoTransform.thatArg

  def do_+ (that: T)(implicit sourceInfo: SourceInfo): T

  /** Outputs the product of `this` and `b`. The resulting width is the sum of
    * the operands.
    *
    * @note can generate a single-cycle multiplier, which can result in
    * significant cycle time and area costs
    */
  final def * (that: T): T = macro SourceInfoTransform.thatArg

  def do_* (that: T)(implicit sourceInfo: SourceInfo): T

  /** Outputs the quotient of `this` and `b`.
    *
    * TODO: full rules
    */
  final def / (that: T): T = macro SourceInfoTransform.thatArg

  def do_/ (that: T)(implicit sourceInfo: SourceInfo): T

  final def % (that: T): T = macro SourceInfoTransform.thatArg

  def do_% (that: T)(implicit sourceInfo: SourceInfo): T

  /** Outputs the difference of `this` and `b`. The resulting width is the max
   *  of the operands plus 1 (should not overflow).
    */
  final def - (that: T): T = macro SourceInfoTransform.thatArg

  def do_- (that: T)(implicit sourceInfo: SourceInfo): T

  /** Outputs true if `this` < `b`.
    */
  final def < (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_< (that: T)(implicit sourceInfo: SourceInfo): Bool

  /** Outputs true if `this` <= `b`.
    */
  final def <= (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_<= (that: T)(implicit sourceInfo: SourceInfo): Bool

  /** Outputs true if `this` > `b`.
    */
  final def > (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_> (that: T)(implicit sourceInfo: SourceInfo): Bool

  /** Outputs true if `this` >= `b`.
    */
  final def >= (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_>= (that: T)(implicit sourceInfo: SourceInfo): Bool

  /** Outputs the minimum of `this` and `b`. The resulting width is the max of
    * the operands.
    */
  final def min(that: T): T = macro SourceInfoTransform.thatArg

  def do_min(that: T)(implicit sourceInfo: SourceInfo): T =
    Mux(this < that, this.asInstanceOf[T], that)

  /** Outputs the maximum of `this` and `b`. The resulting width is the max of
    * the operands.
    */
  final def max(that: T): T = macro SourceInfoTransform.thatArg

  def do_max(that: T)(implicit sourceInfo: SourceInfo): T =
    Mux(this < that, that, this.asInstanceOf[T])
}

/** A data type for unsigned integers, represented as a binary bitvector.
  * Defines arithmetic operations between other integer types.
  */
sealed class UInt private[Chisel] (dir: Direction, width: Width, lit: Option[ULit] = None)
    extends Bits(dir, width, lit) with Num[UInt] {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type =
    new UInt(dir, w).asInstanceOf[this.type]
  private[Chisel] def toType = s"UInt$width"

  override private[Chisel] def fromInt(value: BigInt, width: Int): this.type =
    UInt(value, width).asInstanceOf[this.type]

  override def := (that: Data)(implicit sourceInfo: SourceInfo): Unit = that match {
    case _: UInt => this connect that
    case _ => this badConnect that
  }

  // TODO: refactor to share documentation with Num or add independent scaladoc
  final def unary_- (): UInt = macro SourceInfoTransform.noArg
  final def unary_-% (): UInt = macro SourceInfoTransform.noArg

  def do_unary_- (implicit sourceInfo: SourceInfo) : UInt = UInt(0) - this
  def do_unary_-% (implicit sourceInfo: SourceInfo): UInt = UInt(0) -% this

  override def do_+ (that: UInt)(implicit sourceInfo: SourceInfo): UInt = this +% that
  override def do_- (that: UInt)(implicit sourceInfo: SourceInfo): UInt = this -% that
  override def do_/ (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width), DivideOp, that)
  override def do_% (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width), RemOp, that)
  override def do_* (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width + that.width), TimesOp, that)

  final def * (that: SInt): SInt = macro SourceInfoTransform.thatArg
  def do_* (that: SInt)(implicit sourceInfo: SourceInfo): SInt = that * this

  final def +& (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def +% (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def -& (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def -% (that: UInt): UInt = macro SourceInfoTransform.thatArg

  def do_+& (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt((this.width max that.width) + 1), AddOp, that)
  def do_+% (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this +& that).tail(1)
  def do_-& (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt((this.width max that.width) + 1), SubOp, that)
  def do_-% (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this -& that).tail(1)

  final def & (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def | (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def ^ (that: UInt): UInt = macro SourceInfoTransform.thatArg

  def do_& (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that)
  def do_| (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that)
  def do_^ (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that)

  /** Returns this wire bitwise-inverted. */
  def do_unary_~ (implicit sourceInfo: SourceInfo): UInt =
    unop(sourceInfo, UInt(width = width), BitNotOp)

  // REVIEW TODO: Can this be defined on Bits?
  final def orR(): Bool = macro SourceInfoTransform.noArg
  final def andR(): Bool = macro SourceInfoTransform.noArg
  final def xorR(): Bool = macro SourceInfoTransform.noArg

  def do_orR(implicit sourceInfo: SourceInfo): Bool = this != UInt(0)
  def do_andR(implicit sourceInfo: SourceInfo): Bool = ~this === UInt(0)
  def do_xorR(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, XorReduceOp)

  override def do_< (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterEqOp, that)

  final def != (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def === (that: UInt): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, EqualOp, that)

  final def unary_! () : Bool = macro SourceInfoTransform.noArg

  def do_unary_! (implicit sourceInfo: SourceInfo) : Bool = this === Bits(0)

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, that)
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.shiftRight(that)), ShiftRightOp, that)
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    this >> that.toInt
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width), DynamicShiftRightOp, that)

  final def bitSet(off: UInt, dat: Bool): UInt = macro UIntTransform.bitset

  def do_bitSet(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo): UInt = {
    val bit = UInt(1, 1) << off
    Mux(dat, this | bit, ~(~this | bit))
  }

  /** Returns this UInt as a [[SInt]] with an additional zero in the MSB.
    */
  // TODO: this eventually will be renamed as toSInt, once the existing toSInt
  // completes its deprecation phase.
  final def zext(): SInt = macro SourceInfoTransform.noArg
  def do_zext(implicit sourceInfo: SourceInfo): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width + 1), ConvertOp, ref))

  /** Returns this UInt as a [[SInt]], without changing width or bit value. The
    * SInt is not guaranteed to have the same value (for example, if the MSB is
    * high, it will be interpreted as a negative value).
    */
  override def do_asSInt(implicit sourceInfo: SourceInfo): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width), AsSIntOp, ref))
  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = this
}

// This is currently a factory because both Bits and UInt inherit it.
private[Chisel] sealed trait UIntFactory {
  /** Create a UInt type with inferred width. */
  def apply(): UInt = apply(NO_DIR, Width())
  /** Create a UInt type or port with fixed width. */
  def apply(dir: Direction = NO_DIR, width: Int): UInt = apply(dir, Width(width))
  /** Create a UInt port with inferred width. */
  def apply(dir: Direction): UInt = apply(dir, Width())

  /** Create a UInt literal with inferred width. */
  def apply(value: BigInt): UInt = apply(value, Width())
  /** Create a UInt literal with fixed width. */
  def apply(value: BigInt, width: Int): UInt = apply(value, Width(width))
  /** Create a UInt literal with inferred width. */
  def apply(n: String): UInt = apply(parse(n), parsedWidth(n))
  /** Create a UInt literal with fixed width. */
  def apply(n: String, width: Int): UInt = apply(parse(n), width)

  /** Create a UInt type with specified width. */
  def apply(width: Width): UInt = apply(NO_DIR, width)
  /** Create a UInt port with specified width. */
  def apply(dir: Direction, width: Width): UInt = new UInt(dir, width)
  /** Create a UInt literal with specified width. */
  def apply(value: BigInt, width: Width): UInt = {
    val lit = ULit(value, width)
    new UInt(NO_DIR, lit.width, Some(lit))
  }

  private def parse(n: String) = {
    val (base, num) = n.splitAt(1)
    val radix = base match {
      case "x" | "h" => 16
      case "d" => 10
      case "o" => 8
      case "b" => 2
      case _ => Builder.error(s"Invalid base $base"); 2
    }
    BigInt(num.filterNot(_ == '_'), radix)
  }

  private def parsedWidth(n: String) =
    if (n(0) == 'b') {
      Width(n.length-1)
    } else if (n(0) == 'h') {
      Width((n.length-1) * 4)
    } else {
      Width()
    }
}

object UInt extends UIntFactory

sealed class SInt private (dir: Direction, width: Width, lit: Option[SLit] = None)
    extends Bits(dir, width, lit) with Num[SInt] {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type =
    new SInt(dir, w).asInstanceOf[this.type]
  private[Chisel] def toType = s"SInt$width"

  override def := (that: Data)(implicit sourceInfo: SourceInfo): Unit = that match {
    case _: SInt => this connect that
    case _ => this badConnect that
  }

  override private[Chisel] def fromInt(value: BigInt, width: Int): this.type =
    SInt(value, width).asInstanceOf[this.type]

  final def unary_- (): SInt = macro SourceInfoTransform.noArg
  final def unary_-% (): SInt = macro SourceInfoTransform.noArg

  def unary_- (implicit sourceInfo: SourceInfo): SInt = SInt(0) - this
  def unary_-% (implicit sourceInfo: SourceInfo): SInt = SInt(0) -% this

  /** add (default - no growth) operator */
  override def do_+ (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    this +% that
  /** subtract (default - no growth) operator */
  override def do_- (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    this -% that
  override def do_* (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + that.width), TimesOp, that)
  override def do_/ (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width), DivideOp, that)
  override def do_% (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width), RemOp, that)

  final def * (that: UInt): SInt = macro SourceInfoTransform.thatArg
  def do_* (that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + that.width), TimesOp, that)

  /** add (width +1) operator */
  final def +& (that: SInt): SInt = macro SourceInfoTransform.thatArg
  /** add (no growth) operator */
  final def +% (that: SInt): SInt = macro SourceInfoTransform.thatArg
  /** subtract (width +1) operator */
  final def -& (that: SInt): SInt = macro SourceInfoTransform.thatArg
  /** subtract (no growth) operator */
  final def -% (that: SInt): SInt = macro SourceInfoTransform.thatArg

  def do_+& (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), AddOp, that)
  def do_+% (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    (this +& that).tail(1).asSInt
  def do_-& (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that)
  def do_-% (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    (this -& that).tail(1).asSInt

  final def & (that: SInt): SInt = macro SourceInfoTransform.thatArg
  final def | (that: SInt): SInt = macro SourceInfoTransform.thatArg
  final def ^ (that: SInt): SInt = macro SourceInfoTransform.thatArg

  def do_& (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that).asSInt
  def do_| (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that).asSInt
  def do_^ (that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that).asSInt

  /** Returns this wire bitwise-inverted. */
  def do_unary_~ (implicit sourceInfo: SourceInfo): SInt =
    unop(sourceInfo, UInt(width = width), BitNotOp).asSInt

  override def do_< (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterEqOp, that)

  final def != (that: SInt): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: SInt): Bool = macro SourceInfoTransform.thatArg
  final def === (that: SInt): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, EqualOp, that)

  final def abs(): UInt = macro SourceInfoTransform.noArg

  def do_abs(implicit sourceInfo: SourceInfo): UInt = Mux(this < SInt(0), (-this).asUInt, this.asUInt)

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + that), ShiftLeftOp, that)
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo): SInt =
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width.shiftRight(that)), ShiftRightOp, that)
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo): SInt =
    this >> that.toInt
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width), DynamicShiftRightOp, that)

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
  override def do_asSInt(implicit sourceInfo: SourceInfo): SInt = this
}

object SInt {
  /** Create an SInt type with inferred width. */
  def apply(): SInt = apply(NO_DIR, Width())
  /** Create an SInt type or port with fixed width. */
  def apply(dir: Direction = NO_DIR, width: Int): SInt = apply(dir, Width(width))
  /** Create an SInt port with inferred width. */
  def apply(dir: Direction): SInt = apply(dir, Width())

  /** Create an SInt literal with inferred width. */
  def apply(value: BigInt): SInt = apply(value, Width())
  /** Create an SInt literal with fixed width. */
  def apply(value: BigInt, width: Int): SInt = apply(value, Width(width))

  /** Create an SInt type with specified width. */
  def apply(width: Width): SInt = new SInt(NO_DIR, width)
  /** Create an SInt port with specified width. */
  def apply(dir: Direction, width: Width): SInt = new SInt(dir, width)
  /** Create an SInt literal with specified width. */
  def apply(value: BigInt, width: Width): SInt = {
    val lit = SLit(value, width)
    new SInt(NO_DIR, lit.width, Some(lit))
  }
}

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  */
sealed class Bool(dir: Direction, lit: Option[ULit] = None) extends UInt(dir, Width(1), lit) {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type = {
    require(!w.known || w.get == 1)
    new Bool(dir).asInstanceOf[this.type]
  }

  override private[Chisel] def fromInt(value: BigInt, width: Int): this.type = {
    require((value == 0 || value == 1) && width == 1)
    Bool(value == 1).asInstanceOf[this.type]
  }

  // REVIEW TODO: Why does this need to exist and have different conventions
  // than Bits?
  final def & (that: Bool): Bool = macro SourceInfoTransform.thatArg
  final def | (that: Bool): Bool = macro SourceInfoTransform.thatArg
  final def ^ (that: Bool): Bool = macro SourceInfoTransform.thatArg

  def do_& (that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitAndOp, that)
  def do_| (that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitOrOp, that)
  def do_^ (that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitXorOp, that)

  /** Returns this wire bitwise-inverted. */
  override def do_unary_~ (implicit sourceInfo: SourceInfo): Bool =
    unop(sourceInfo, Bool(), BitNotOp)

  /** Outputs the logical OR of two Bools.
   */
  def || (that: Bool): Bool = macro SourceInfoTransform.thatArg

  def do_|| (that: Bool)(implicit sourceInfo: SourceInfo): Bool = this | that

  /** Outputs the logical AND of two Bools.
   */
  def && (that: Bool): Bool = macro SourceInfoTransform.thatArg

  def do_&& (that: Bool)(implicit sourceInfo: SourceInfo): Bool = this & that
}

object Bool {
  /** Creates an empty Bool.
   */
  def apply(dir: Direction = NO_DIR): Bool = new Bool(dir)

  /** Creates Bool literal.
   */
  def apply(x: Boolean): Bool = new Bool(NO_DIR, Some(ULit(if (x) 1 else 0, Width(1))))
}

object Mux {
  /** Creates a mux, whose output is one of the inputs depending on the
    * value of the condition.
    *
    * @param cond condition determining the input to choose
    * @param con the value chosen when `cond` is true
    * @param alt the value chosen when `cond` is false
    * @example
    * {{{
    * val muxOut = Mux(data_in === UInt(3), UInt(3, 4), UInt(0, 4))
    * }}}
    */
  def apply[T <: Data](cond: Bool, con: T, alt: T): T = macro MuxTransform.apply[T]

  def do_apply[T <: Data](cond: Bool, con: T, alt: T)(implicit sourceInfo: SourceInfo): T =
    (con, alt) match {
    // Handle Mux(cond, UInt, Bool) carefully so that the concrete type is UInt
    case (c: Bool, a: Bool) => doMux(cond, c, a).asInstanceOf[T]
    case (c: UInt, a: Bool) => doMux(cond, c, a << 0).asInstanceOf[T]
    case (c: Bool, a: UInt) => doMux(cond, c << 0, a).asInstanceOf[T]
    case (c: Bits, a: Bits) => doMux(cond, c, a).asInstanceOf[T]
    case _ => doAggregateMux(cond, con, alt)
  }

  private def doMux[T <: Data](cond: Bool, con: T, alt: T)(implicit sourceInfo: SourceInfo): T = {
    require(con.getClass == alt.getClass, s"can't Mux between ${con.getClass} and ${alt.getClass}")
    val d = alt.cloneTypeWidth(con.width max alt.width)
    pushOp(DefPrim(sourceInfo, d, MultiplexOp, cond.ref, con.ref, alt.ref))
  }

  private def doAggregateMux[T <: Data](cond: Bool, con: T, alt: T)(implicit sourceInfo: SourceInfo): T = {
    require(con.getClass == alt.getClass, s"can't Mux between ${con.getClass} and ${alt.getClass}")
    for ((c, a) <- con.flatten zip alt.flatten)
      require(c.width == a.width, "can't Mux between aggregates of different width")
    doMux(cond, con, alt)
  }
}

