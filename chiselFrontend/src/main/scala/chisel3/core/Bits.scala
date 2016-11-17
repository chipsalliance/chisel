// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, DeprecatedSourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform,
  UIntTransform, MuxTransform}
import chisel3.internal.firrtl.PrimOp._
// TODO: remove this once we have CompileOptions threaded through the macro system.
import chisel3.core.ExplicitCompileOptions.NotStrict

/** Element is a leaf data type: it cannot contain other Data objects. Example
  * uses are for representing primitive data types, like integers and bits.
  */
abstract class Element(private[core] val width: Width) extends Data {
  /**
   * Elements can actually be bound to the hardware graph and thus must store
   * that binding information.
   */
  private[this] var _binding: Binding = UnboundBinding(None)
  // Define setter/getter pairing
  // Can only bind something that has not yet been bound.
  private[core] def binding_=(target: Binding): Unit = _binding match {
    case UnboundBinding(_) => {
      _binding = target
      _binding
    }
    case _ => throw Binding.AlreadyBoundException(_binding.toString)
      // Other checks should have caught this.
  }
  private[core] def binding = _binding

  /** Return the binding for some bits. */
  def dir: Direction = binding.direction.getOrElse(Direction.Unspecified)

  private[chisel3] final def allElements: Seq[Element] = Seq(this)
  def widthKnown: Boolean = width.known
  def name: String = getRef.name

  private[core] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    pushCommand(Connect(sourceInfo, this.lref, that.ref))
}

/** A data type for values represented by a single bitvector. Provides basic
  * bitwise operations.
  */
sealed abstract class Bits(width: Width, override val litArg: Option[LitArg])
    extends Element(width) {
  // TODO: perhaps make this concrete?
  // Arguments for: self-checking code (can't do arithmetic on bits)
  // Arguments against: generates down to a FIRRTL UInt anyways

  private[chisel3] def fromInt(x: BigInt, w: Int): this.type

  private[chisel3] def flatten: IndexedSeq[Bits] = IndexedSeq(this)

  def cloneType: this.type = cloneTypeWidth(width)

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
    binop(sourceInfo, UInt(Width(n)), HeadOp, n)
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
      (((litValue() >> x.toInt) & 1) == 1).asBool
    } else {
      Binding.checkSynthesizable(this, s"'this' ($this)")
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
      ((litValue >> y) & ((BigInt(1) << w) - 1)).asUInt(w.W)
    } else {
      Binding.checkSynthesizable(this, s"'this' ($this)")
      pushOp(DefPrim(sourceInfo, UInt(Width(w)), BitsExtractOp, this.ref, ILit(x), ILit(y)))
    }
  }

  // REVIEW TODO: again, is this necessary? Or just have this and use implicits?
  final def apply(x: BigInt, y: BigInt): UInt = macro SourceInfoTransform.xyArg

  final def do_apply(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    apply(x.toInt, y.toInt)

  private[core] def unop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp): T = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref))
  }
  private[core] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: BigInt): T = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, ILit(other)))
  }
  private[core] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: Bits): T = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(other, s"'other' ($other)")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, other.ref))
  }
  private[core] def compop(sourceInfo: SourceInfo, op: PrimOp, other: Bits): Bool = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(other, s"'other' ($other)")
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }
  private[core] def redop(sourceInfo: SourceInfo, op: PrimOp): Bool = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref))
  }

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

  /** Reinterpret cast as a FixedPoint.
    *
    * @note value not guaranteed to be preserved: for example, an UInt of width
    * 3 and value 7 (0b111) would become a FixedInt with value -1, the interpretation
    * of the number is also affected by the specified binary point.  Caution advised
    */
  final def asFixedPoint(that: BinaryPoint): FixedPoint = macro SourceInfoTransform.thatArg

  def do_asFixedPoint(that: BinaryPoint)(implicit sourceInfo: SourceInfo): FixedPoint = {
    throwException(s"Cannot call .asFixedPoint on $this")
  }

  /** Reinterpret cast to Bits. */
  @deprecated("Use asUInt, which does the same thing but returns a more concrete type", "chisel3")
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

  override def do_fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = {
    val res = Wire(this, null).asInstanceOf[this.type]
    res := that
    res
  }

  /** Default print as [[Decimal]] */
  final def toPrintable: Printable = Decimal(this)
}

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
sealed class UInt private[core] (width: Width, lit: Option[ULit] = None)
    extends Bits(width, lit) with Num[UInt] {

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new UInt(w).asInstanceOf[this.type]
  private[chisel3] def toType = s"UInt$width"

  override private[chisel3] def fromInt(value: BigInt, width: Int): this.type =
    value.asUInt(width.W).asInstanceOf[this.type]

  // TODO: refactor to share documentation with Num or add independent scaladoc
  final def unary_- (): UInt = macro SourceInfoTransform.noArg
  final def unary_-% (): UInt = macro SourceInfoTransform.noArg

  def do_unary_- (implicit sourceInfo: SourceInfo) : UInt = 0.U - this
  def do_unary_-% (implicit sourceInfo: SourceInfo): UInt = 0.U -% this

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

  def do_orR(implicit sourceInfo: SourceInfo): Bool = this != 0.U
  def do_andR(implicit sourceInfo: SourceInfo): Bool = ~this === 0.U
  def do_xorR(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, XorReduceOp)

  override def do_< (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterEqOp, that)

  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
  final def != (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def === (that: UInt): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: UInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, EqualOp, that)

  final def unary_! () : Bool = macro SourceInfoTransform.noArg

  def do_unary_! (implicit sourceInfo: SourceInfo) : Bool = this === 0.U(1)

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
    val bit = 1.U(1) << off
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
  override def do_asFixedPoint(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo): FixedPoint = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
    }
  }
}

// This is currently a factory because both Bits and UInt inherit it.
trait UIntFactory {
  /** Create a UInt type with inferred width. */
  def apply(): UInt = apply(Width())
  /** Create a UInt port with specified width. */
  def apply(width: Width): UInt = new UInt(width)

   /** Create a UInt literal with specified width. */
  protected[chisel3] def Lit(value: BigInt, width: Width): UInt = {
    val lit = ULit(value, width)
    val result = new UInt(lit.width, Some(lit))
    // Bind result to being an Literal
    result.binding = LitBinding()
    result
  }

  /** Create a UInt with the specified range */
  def apply(range: Range): UInt = {
    apply(range.getWidth)
  }
  /** Create a UInt with the specified range */
  def apply(range: (NumericBound[Int], NumericBound[Int])): UInt = {
    apply(KnownUIntRange(range._1, range._2))
  }
}

object UInt extends UIntFactory
object Bits extends UIntFactory

sealed class SInt private[core] (width: Width, lit: Option[SLit] = None)
    extends Bits(width, lit) with Num[SInt] {

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new SInt(w).asInstanceOf[this.type]
  private[chisel3] def toType = s"SInt$width"

  override private[chisel3] def fromInt(value: BigInt, width: Int): this.type =
    value.asSInt(width.W).asInstanceOf[this.type]

  final def unary_- (): SInt = macro SourceInfoTransform.noArg
  final def unary_-% (): SInt = macro SourceInfoTransform.noArg

  def unary_- (implicit sourceInfo: SourceInfo): SInt = 0.S - this
  def unary_-% (implicit sourceInfo: SourceInfo): SInt = 0.S -% this

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

  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
  final def != (that: SInt): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: SInt): Bool = macro SourceInfoTransform.thatArg
  final def === (that: SInt): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: SInt)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, EqualOp, that)

  final def abs(): UInt = macro SourceInfoTransform.noArg

  def do_abs(implicit sourceInfo: SourceInfo): UInt = Mux(this < 0.S, (-this).asUInt, this.asUInt)

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
  override def do_asFixedPoint(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo): FixedPoint = {
    pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref))
  }
}

trait SIntFactory {
  /** Create an SInt type with inferred width. */
  def apply(): SInt = apply(Width())
  /** Create a SInt type or port with fixed width. */
  def apply(width: Width): SInt = new SInt(width)

  /** Create a SInt with the specified range */
  def apply(range: Range): SInt = {
    apply(range.getWidth)
  }
  /** Create a SInt with the specified range */
  def apply(range: (NumericBound[Int], NumericBound[Int])): SInt = {
    apply(KnownSIntRange(range._1, range._2))
  }

   /** Create an SInt literal with specified width. */
  protected[chisel3] def Lit(value: BigInt, width: Width): SInt = {
    val lit = SLit(value, width)
    val result = new SInt(lit.width, Some(lit))
    // Bind result to being an Literal
    result.binding = LitBinding()
    result
  }
}

object SInt extends SIntFactory

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  */
sealed class Bool(lit: Option[ULit] = None) extends UInt(Width(1), lit) {
  private[core] override def cloneTypeWidth(w: Width): this.type = {
    require(!w.known || w.get == 1)
    new Bool().asInstanceOf[this.type]
  }

  override private[chisel3] def fromInt(value: BigInt, width: Int): this.type = {
    require((value == 0 || value == 1) && width == 1)
    (value == 1).asBool.asInstanceOf[this.type]
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

  /** Reinterprets this Bool as a Clock.  */
  def asClock(): Clock = macro SourceInfoTransform.noArg

  def do_asClock(implicit sourceInfo: SourceInfo): Clock = pushOp(DefPrim(sourceInfo, Clock(), AsClockOp, ref))
}

trait BoolFactory {
  /** Creates an empty Bool.
   */
  def apply(): Bool = new Bool()

  /** Creates Bool literal.
   */
  protected[chisel3] def Lit(x: Boolean): Bool = {
    val result = new Bool(Some(ULit(if (x) 1 else 0, Width(1))))
    // Bind result to being an Literal
    result.binding = LitBinding()
    result
  }
}

object Bool extends BoolFactory

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
    Binding.checkSynthesizable(cond, s"'cond' ($cond)")
    Binding.checkSynthesizable(con, s"'con' ($con)")
    Binding.checkSynthesizable(alt, s"'alt' ($alt)")
    val d = alt.cloneTypeWidth(con.width max alt.width)
    pushOp(DefPrim(sourceInfo, d, MultiplexOp, cond.ref, con.ref, alt.ref))
  }

  private[core] def typesCompatible[T <: Data](x: T, y: T): Boolean = {
    val sameTypes = x.getClass == y.getClass
    val sameElements = x.flatten zip y.flatten forall { case (a, b) => a.getClass == b.getClass && a.width == b.width }
    val sameNumElements = x.flatten.size == y.flatten.size
    sameTypes && sameElements && sameNumElements
  }

  private def doAggregateMux[T <: Data](cond: Bool, con: T, alt: T)(implicit sourceInfo: SourceInfo): T = {
    require(typesCompatible(con, alt), s"can't Mux between heterogeneous types ${con.getClass} and ${alt.getClass}")
    doMux(cond, con, alt)
  }
}

//scalastyle:off number.of.methods
/**
  * A sealed class representing a fixed point number that has a bit width and a binary point
  * The width and binary point may be inferred.
  *
  * IMPORTANT: The API provided here is experimental and may change in the future.
  *
  * @param width       bit width of the fixed point number
  * @param binaryPoint the position of the binary point with respect to the right most bit of the width
  *                    currently this should be positive but it is hoped to soon support negative points
  *                    and thus use this field as a simple exponent
  * @param lit
  */
sealed class FixedPoint private (width: Width, val binaryPoint: BinaryPoint, lit: Option[FPLit] = None)
    extends Bits(width, lit) with Num[FixedPoint] {
  private[core] override def cloneTypeWidth(w: Width): this.type =
    new FixedPoint(w, binaryPoint).asInstanceOf[this.type]
  private[chisel3] def toType = s"Fixed$width$binaryPoint"

  def := (that: Data)(implicit sourceInfo: SourceInfo): Unit = that match {
    case _: FixedPoint => this connect that
    case _ => this badConnect that
  }

  private[chisel3] def fromInt(value: BigInt, width: Int): this.type = {
    throwException(s"Don't use $this.fromInt($value, $width): Use literal constructors instead")
  }

  final def unary_- (): FixedPoint = macro SourceInfoTransform.noArg
  final def unary_-% (): FixedPoint = macro SourceInfoTransform.noArg

  def unary_- (implicit sourceInfo: SourceInfo): FixedPoint = FixedPoint.fromBigInt(0) - this
  def unary_-% (implicit sourceInfo: SourceInfo): FixedPoint = FixedPoint.fromBigInt(0) -% this

  /** add (default - no growth) operator */
  override def do_+ (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    this +% that
  /** subtract (default - no growth) operator */
  override def do_- (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    this -% that
  override def do_* (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that.width, this.binaryPoint + that.binaryPoint), TimesOp, that)
  override def do_/ (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    throwException(s"division is illegal on FixedPoint types")
  override def do_% (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    throwException(s"mod is illegal on FixedPoint types")

  final def * (that: UInt): FixedPoint = macro SourceInfoTransform.thatArg
  def do_* (that: UInt)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that.width, binaryPoint), TimesOp, that)

  final def * (that: SInt): FixedPoint = macro SourceInfoTransform.thatArg
  def do_* (that: SInt)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that.width, binaryPoint), TimesOp, that)

  /** add (width +1) operator */
  final def +& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  /** add (no growth) operator */
  final def +% (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  /** subtract (width +1) operator */
  final def -& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  /** subtract (no growth) operator */
  final def -% (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  def do_+& (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint((this.width max that.width) + 1, this.binaryPoint max that.binaryPoint), AddOp, that)
  def do_+% (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    (this +& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)
  def do_-& (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint((this.width max that.width) + 1, this.binaryPoint max that.binaryPoint), SubOp, that)
  def do_-% (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    (this -& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)

  final def & (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  final def | (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  final def ^ (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  def do_& (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    throwException(s"And is illegal between $this and $that")
  def do_| (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    throwException(s"Or is illegal between $this and $that")
  def do_^ (that: FixedPoint)(implicit sourceInfo: SourceInfo): FixedPoint =
    throwException(s"Xor is illegal between $this and $that")

  final def setBinaryPoint(that: Int): FixedPoint = macro SourceInfoTransform.thatArg

  def do_setBinaryPoint(that: Int)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width, KnownBinaryPoint(that)), SetBinaryPoint, that)

  /** Returns this wire bitwise-inverted. */
  def do_unary_~ (implicit sourceInfo: SourceInfo): FixedPoint =
    throwException(s"Not is illegal on $this")

  // TODO(chick): Consider comparison with UInt and SInt
  override def do_< (that: FixedPoint)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: FixedPoint)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: FixedPoint)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: FixedPoint)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, GreaterEqOp, that)

  final def != (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg
  final def === (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: FixedPoint)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: FixedPoint)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: FixedPoint)(implicit sourceInfo: SourceInfo): Bool = compop(sourceInfo, EqualOp, that)

  final def abs(): UInt = macro SourceInfoTransform.noArg

  def do_abs(implicit sourceInfo: SourceInfo): UInt = {
    Mux(this < FixedPoint.fromBigInt(0), (FixedPoint.fromBigInt(0)-this).asUInt, this.asUInt)
  }

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that, this.binaryPoint), ShiftLeftOp, that)
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo): FixedPoint =
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width.dynamicShiftLeft(that.width), this.binaryPoint), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width.shiftRight(that), this.binaryPoint), ShiftRightOp, that)
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo): FixedPoint =
    this >> that.toInt
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width, this.binaryPoint), DynamicShiftRightOp, that)

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
  override def do_asSInt(implicit sourceInfo: SourceInfo): SInt = pushOp(DefPrim(sourceInfo, SInt(this.width), AsSIntOp, ref))
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
  /** Create an FixedPoint type with inferred width. */
  def apply(): FixedPoint = apply(Width(), BinaryPoint())

  /** Create an FixedPoint type or port with fixed width. */
  def apply(width: Int, binaryPoint: Int): FixedPoint = apply(Width(width), BinaryPoint(binaryPoint))
  /** Create an FixedPoint port with inferred width. */
  def apply(dir: Direction): FixedPoint = apply(dir, Width(), BinaryPoint())

  /** Create an FixedPoint literal with inferred width from BigInt.
    * Use PrivateObject to force users to specify width and binaryPoint by name
    */
  def fromBigInt(value: BigInt, width: Int = -1, binaryPoint: Int = 0): FixedPoint =
    if(width == -1) {
      apply(value, Width(), BinaryPoint(binaryPoint))
    }
    else {
      apply(value, Width(width), BinaryPoint(binaryPoint))
    }
  /** Create an FixedPoint literal with inferred width from Double.
    * Use PrivateObject to force users to specify width and binaryPoint by name
    */
  def fromDouble(value: Double, dummy: PrivateType = PrivateObject,
                 width: Int = -1, binaryPoint: Int = 0): FixedPoint = {
    fromBigInt(
      toBigInt(value, binaryPoint), width = width, binaryPoint = binaryPoint
    )
  }

  /** Create an FixedPoint type with specified width and binary position. */
  def apply(width: Width, binaryPoint: BinaryPoint): FixedPoint = new FixedPoint(width, binaryPoint)
  /** Create an FixedPoint port with specified width and binary position. */
  def apply(dir: Direction, width: Width, binaryPoint: BinaryPoint): FixedPoint = new FixedPoint(width, binaryPoint)
  def apply(value: BigInt, width: Width, binaryPoint: BinaryPoint): FixedPoint = {
    val lit = FPLit(value, width, binaryPoint)
    val newLiteral = new FixedPoint(lit.width, lit.binaryPoint, Some(lit))
    newLiteral.binding = LitBinding()
    newLiteral
  }

  /**
    * How to create a bigint from a double with a specific binaryPoint
    * @param x               a double value
    * @param binaryPoint     a binaryPoint that you would like to use
    * @return
    */
  def toBigInt(x: Double, binaryPoint    : Int): BigInt = {
    val multiplier = math.pow(2,binaryPoint    )
    val result = BigInt(math.round(x * multiplier))
    result
  }

  /**
    * converts a bigInt with the given binaryPoint into the double representation
    * @param i            a bigint
    * @param binaryPoint  the implied binaryPoint of @i
    * @return
    */
  def toDouble(i: BigInt, binaryPoint    : Int): Double = {
    val multiplier = math.pow(2,binaryPoint    )
    val result = i.toDouble / multiplier
    result
  }

}
