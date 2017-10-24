// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros
import collection.mutable
import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{DeprecatedSourceInfo, SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform, UIntTransform}
import chisel3.internal.firrtl.PrimOp._
import _root_.firrtl.ir.{Closed, Open, UnknownBound, Bound}
import _root_.firrtl.passes.IsKnown
import _root_.firrtl.{ir => firrtlir}

//scalastyle:off method.name

object TypePropagate {
  def apply[T <: Bits](op: firrtlir.PrimOp, args: Seq[Bits], consts: Seq[Int]): T = {
    val expArgs = args.map(toFirrtlType).map(_root_.firrtl.WRef("whatever", _, _root_.firrtl.ExpKind, _root_.firrtl.UNKNOWNGENDER))
    val prim = firrtlir.DoPrim(op, expArgs, consts.map(BigInt(_)), firrtlir.UnknownType)
    val tpe = _root_.firrtl.PrimOps.set_primop_type(prim).tpe
    toChiselType(tpe).asInstanceOf[T]
  }
  def chiselWidthToFirrtlWidth(w: Width): firrtlir.Width = w match {
    case chisel3.internal.firrtl.KnownWidth(n) => firrtlir.IntWidth(n)
    case _ => firrtlir.UnknownWidth
  }
  def firrtlWidthToChiselWidth(w: firrtlir.Width): Width  = w match {
    case firrtlir.IntWidth(n) => chisel3.internal.firrtl.KnownWidth(n.toInt)
    case _ => UnknownWidth()
  }
  def toFirrtlType(t: Bits): firrtlir.Type = t match {
    case u: UInt => firrtlir.UIntType(chiselWidthToFirrtlWidth(u.width))
    case u: SInt => firrtlir.SIntType(chiselWidthToFirrtlWidth(u.width))
    case i: Interval => firrtlir.IntervalType(i.range.lowerBound, i.range.upperBound, i.range.firrtlBinaryPoint)
  }
  def toChiselType(t: firrtlir.Type): Bits = t match {
    case u: firrtlir.UIntType => UInt()
    case i: firrtlir.IntervalType => Interval()
  }
}

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
//scalastyle:off number.of.methods
sealed abstract class Bits(width: Width, override val litArg: Option[LitArg])
    extends Element(width) {
  // TODO: perhaps make this concrete?
  // Arguments for: self-checking code (can't do arithmetic on bits)
  // Arguments against: generates down to a FIRRTL UInt anyways

  // Only used for in a few cases, hopefully to be removed
  private[core] def cloneTypeWidth(width: Width): this.type

  def cloneType: this.type = cloneTypeWidth(width)

  final def tail(n: Int): UInt = macro SourceInfoTransform.nArg
  final def head(n: Int): UInt = macro SourceInfoTransform.nArg

  def do_tail(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val w = width match {
      case KnownWidth(x) =>
        require(x >= n, s"Can't tail($n) for width $x < $n")
        Width(x - n)
      case UnknownWidth() => Width()
    }
    binop(sourceInfo, UInt(width = w), TailOp, n)
  }


  def do_head(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
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

  final def do_apply(x: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
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

  final def do_apply(x: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = apply(BigInt(x))

  /** Returns the specified bit on this wire as a [[Bool]], dynamically
    * addressed.
    */
  final def apply(x: UInt): Bool = macro SourceInfoTransform.xArg

  final def do_apply(x: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    val theBits = this >> x
    theBits(0)
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

  final def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
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

  final def do_apply(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
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

  def do_pad(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type =
    binop(sourceInfo, cloneTypeWidth(this.width max Width(that)), PadOp, that)

  /** Returns this wire bitwise-inverted. */
  final def unary_~ (): Bits = macro SourceInfoWhiteboxTransform.noArg

  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits


  /** Shift left operation */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or Bits?
  final def << (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Returns this wire statically left shifted by the specified amount,
    * inserting zeros into the least significant bits.
    *
    * The width of the output is `other` larger than the input.
    */
  final def << (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Returns this wire dynamically left shifted by the specified amount,
    * inserting zeros into the least significant bits.
    *
    * The width of the output is `pow(2, width(other))` larger than the input.
    */
  final def << (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Shift right operation */
  // REVIEW TODO: redundant
  final def >> (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Returns this wire statically right shifted by the specified amount,
    * inserting zeros into the most significant bits.
    *
    * The width of the output is the same as the input.
    */
  final def >> (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Returns this wire dynamically right shifted by the specified amount,
    * inserting zeros into the most significant bits.
    *
    * The width of the output is the same as the input.
    */
  final def >> (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits

  /** Returns the contents of this wire as a [[Vec]] of [[Bool]]s.
    */
  final def toBools(): Seq[Bool] = macro SourceInfoTransform.noArg

  def toBools(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Seq[Bool] =
    Seq.tabulate(this.getWidth)(i => this(i))

  /** Reinterpret cast to a SInt.
    *
    * @note value not guaranteed to be preserved: for example, an UInt of width
    * 3 and value 7 (0b111) would become a SInt with value -1
    */
  final def asSInt(): SInt = macro SourceInfoTransform.noArg

  def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt

  /** Reinterpret cast as a FixedPoint.
    *
    * @note value not guaranteed to be preserved: for example, an UInt of width
    * 3 and value 7 (0b111) would become a FixedInt with value -1, the interpretation
    * of the number is also affected by the specified binary point.  Caution advised
    */
  final def asFixedPoint(that: BinaryPoint): FixedPoint = macro SourceInfoTransform.thatArg

  def do_asFixedPoint(that: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    throwException(s"Cannot call .asFixedPoint on $this")
  }
  /** Reinterpret cast as a Interval.
    *
    * @note value not guaranteed to be preserved: for example, an UInt of width
    * 3 and value 7 (0b111) would become a FixedInt with value -1, the interpretation
    * of the number is also affected by the specified binary point.  Caution advised
    */
  final def asInterval(x: BinaryPoint, y: IntervalRange): Interval = macro SourceInfoTransform.xyArg

  def do_asInterval(binaryPoint: BinaryPoint, range: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    throwException(s"Cannot call .asInterval on $this")
  }

  final def asInterval(x: IntervalRange): Interval = macro SourceInfoTransform.xArg

  def do_asInterval(range: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    throwException(s"Cannot call .asInterval on $this")
  }

  /** Reinterpret cast to Bits. */
  @deprecated("Use asUInt, which does the same thing but returns a more concrete type", "chisel3")
  final def asBits(): Bits = macro SourceInfoTransform.noArg

  def do_asBits(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits = asUInt()

  @deprecated("Use asSInt, which makes the reinterpret cast more explicit", "chisel3")
  final def toSInt(implicit compileOptions: CompileOptions): SInt = do_asSInt(DeprecatedSourceInfo, compileOptions)
  @deprecated("Use asUInt, which makes the reinterpret cast more explicit", "chisel3")
  final def toUInt(implicit compileOptions: CompileOptions): UInt = do_asUInt(DeprecatedSourceInfo, compileOptions)

  final def toBool(): Bool = macro SourceInfoTransform.noArg

  def do_toBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
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

  def do_## (that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val w = this.width + that.width
    pushOp(DefPrim(sourceInfo, UInt(w), ConcatOp, this.ref, that.ref))
  }

  /** Default print as [[Decimal]] */
  final def toPrintable: Printable = Decimal(this)
}

// REVIEW TODO: Further discussion needed on what Num actually is.
/** Abstract trait defining operations available on numeric-like wire data
  * types.
  */
abstract trait Num[T <: Data] {
  self: Num[T] =>
  // def << (b: T): T
  // def >> (b: T): T
  //def unary_-(): T

  // REVIEW TODO: double check ops conventions against FIRRTL

  /** Outputs the sum of `this` and `b`. The resulting width is the max of the
    * operands plus 1 (should not overflow).
    */
  final def + (that: T): T = macro SourceInfoTransform.thatArg

  def do_+ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Outputs the product of `this` and `b`. The resulting width is the sum of
    * the operands.
    *
    * @note can generate a single-cycle multiplier, which can result in
    * significant cycle time and area costs
    */
  final def * (that: T): T = macro SourceInfoTransform.thatArg

  def do_* (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Outputs the quotient of `this` and `b`.
    *
    * TODO: full rules
    */
  final def / (that: T): T = macro SourceInfoTransform.thatArg

  def do_/ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  final def % (that: T): T = macro SourceInfoTransform.thatArg

  def do_% (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Outputs the difference of `this` and `b`. The resulting width is the max
   *  of the operands plus 1 (should not overflow).
    */
  final def - (that: T): T = macro SourceInfoTransform.thatArg

  def do_- (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Outputs true if `this` < `b`.
    */
  final def < (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Outputs true if `this` <= `b`.
    */
  final def <= (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Outputs true if `this` > `b`.
    */
  final def > (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Outputs true if `this` >= `b`.
    */
  final def >= (that: T): Bool = macro SourceInfoTransform.thatArg

  def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Outputs the absolute value of `this`. The resulting width is the unchanged */
  final def abs(): T = macro SourceInfoTransform.noArg
  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Outputs the minimum of `this` and `b`. The resulting width is the max of
    * the operands.
    */
  final def min(that: T): T = macro SourceInfoTransform.thatArg

  def do_min(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    Mux(this < that, this.asInstanceOf[T], that)

  /** Outputs the maximum of `this` and `b`. The resulting width is the max of
    * the operands.
    */
  final def max(that: T): T = macro SourceInfoTransform.thatArg

  def do_max(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    Mux(this < that, that, this.asInstanceOf[T])
}

/** A data type for unsigned integers, represented as a binary bitvector.
  * Defines arithmetic operations between other integer types.
  */
sealed class UInt private[core] (width: Width, lit: Option[ULit] = None)
    extends Bits(width, lit) with Num[UInt] {

  private[core] override def typeEquivalent(that: Data): Boolean =
    that.isInstanceOf[UInt] && this.width == that.width

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new UInt(w).asInstanceOf[this.type]
  private[chisel3] def toType = s"UInt$width"

  // TODO: refactor to share documentation with Num or add independent scaladoc
  final def unary_- (): UInt = macro SourceInfoTransform.noArg
  final def unary_-% (): UInt = macro SourceInfoTransform.noArg

  def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : UInt = 0.U - this
  def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = 0.U -% this

  override def do_+ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this +% that
  override def do_- (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this -% that
  override def do_/ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), DivideOp, that)
  override def do_% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), RemOp, that)
  override def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width + that.width), TimesOp, that)

  final def * (that: SInt): SInt = macro SourceInfoTransform.thatArg
  def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = that * this

  final def +& (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def +% (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def -& (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def -% (that: UInt): UInt = macro SourceInfoTransform.thatArg

  def do_+& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt((this.width max that.width) + 1), AddOp, that)
  def do_+% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this +& that).tail(1)
  def do_-& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that).asUInt
  def do_-% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this -& that).tail(1)

  final def & (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def | (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def ^ (that: UInt): UInt = macro SourceInfoTransform.thatArg

//  override def abs: UInt = macro SourceInfoTransform.noArg
  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this

  def do_& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that)
  def do_| (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that)
  def do_^ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that)

  /** Returns this wire bitwise-inverted. */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    unop(sourceInfo, UInt(width = width), BitNotOp)

  // REVIEW TODO: Can this be defined on Bits?
  final def orR(): Bool = macro SourceInfoTransform.noArg
  final def andR(): Bool = macro SourceInfoTransform.noArg
  final def xorR(): Bool = macro SourceInfoTransform.noArg

  def do_orR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this != 0.U
  def do_andR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = ~this === 0.U
  def do_xorR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = redop(sourceInfo, XorReduceOp)

  override def do_< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
  final def != (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def === (that: UInt): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)

  final def unary_! () : Bool = macro SourceInfoTransform.noArg

  def do_unary_! (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : Bool = this === 0.U(1.W)

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, that)
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.shiftRight(that)), ShiftRightOp, that)
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this >> that.toInt
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), DynamicShiftRightOp, that)

  final def bitSet(off: UInt, dat: Bool): UInt = macro UIntTransform.bitset

  def do_bitSet(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val bit = 1.U(1.W) << off
    Mux(dat, this | bit, ~(~this | bit))
  }

  /** Returns this UInt as a [[SInt]] with an additional zero in the MSB.
    */
  // TODO: this eventually will be renamed as toSInt, once the existing toSInt
  // completes its deprecation phase.
  final def zext(): SInt = macro SourceInfoTransform.noArg
  def do_zext(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width + 1), ConvertOp, ref))

  /** Returns this UInt as a [[SInt]], without changing width or bit value. The
    * SInt is not guaranteed to have the same value (for example, if the MSB is
    * high, it will be interpreted as a negative value).
    */
  override def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width), AsSIntOp, ref))
  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this
  override def do_asFixedPoint(binaryPoint: BinaryPoint)
                              (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(
          s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
    }
  }
  //TODO: (chick) Refactor? -- This code doesn't work b/c Firrlt PrimOp needs 3 args
  def do_asInterval(binaryPoint: BinaryPoint)
                            (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        /*pushOp(DefPrim(sourceInfo,
          Interval(width,
            new IntervalRange(UnknownBound, UnknownBound, IntervalRange.getBinaryPoint(binaryPoint))), AsIntervalOp, ref))
        */
        throwException("asInterval INVALID")
      case _ =>
        throwException(
          s"cannot call $this.asInterval($binaryPoint), you must specify a known binaryPoint")
    }
  }

  override def do_asInterval(range: IntervalRange = IntervalRange.unknownRange)
                            (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {                       
    (range.lower, range.upper, range.binaryPoint) match {
      case (lx: IsKnown, ux: IsKnown, KnownBinaryPoint(bp)) =>
        // No mechanism to pass open/close to firrtl so need to handle directly
        // TODO: (chick) can we pass open/close to firrtl?
        val l = lx match {
          case Open(x) => x + math.pow(2, -bp)
          case Closed(x) => x
        }
        val u = ux match { 
          case Open(x) => x - math.pow(2, -bp)
          case Closed(x) => x
        }
        //TODO: (chick) Need to determine, what asInterval needs, and why it might need min and max as args -- CAN IT BE UNKNOWN?
        // Angie's operation: Decimal -> Int -> Decimal loses information. Need to be conservative here?
        val minBI = (l * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        val maxBI = (u * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.CEILING).toBigIntExact.get
        pushOp(DefPrim(sourceInfo, Interval(width, range), AsIntervalOp, ref, ILit(minBI), ILit(maxBI), ILit(bp)))
      case _ =>
        throwException(
          s"cannot call $this.asInterval($range), you must specify a known binaryPoint and range")
    }
  }
  def do_fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = {
    val res = Wire(this, null).asInstanceOf[this.type]
    res := (that match {
      case u: UInt => u
      case _ => that.asUInt
    })
    res
  }
  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that.asUInt
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

  /** Create a UInt with the specified range
    * Because the ranges passed in are based on the bit widths of SInts which are the underlying type of
    * interval we need to subtract 1 here before we create the UInt
    */
  def apply(range: IntervalRange): UInt = {
    range.lower match {
      case Closed(n) =>
        if(n < 0) throw new IllegalArgumentException(s"Creating UInt: Invalid range with ${range.serialize}")
      case Open(n) =>
        if(n <= 0) throw new IllegalArgumentException(s"Creating UInt: Invalid range with ${range.serialize}")
      case _ =>
    }
    val uIntWidth = range.getWidth match {
      case KnownWidth(n) => KnownWidth((n - 1).max(0))
      case unknownWidth: UnknownWidth => unknownWidth
    }
    apply(uIntWidth)
  }
//TODO: Fix this later
//  /** Create a UInt with the specified range */
//  def apply(range: (NumericBound[Int], NumericBound[Int])): UInt = {
//    apply(KnownUIntRange(range._1, range._2))
//  }
}

object UInt extends UIntFactory
object Bits extends UIntFactory

sealed class SInt private[core] (width: Width, lit: Option[SLit] = None)
    extends Bits(width, lit) with Num[SInt] {

  private[core] override def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass && this.width == that.width  // TODO: should this be true for unspecified widths?

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new SInt(w).asInstanceOf[this.type]
  private[chisel3] def toType = s"SInt$width"

  final def unary_- (): SInt = macro SourceInfoTransform.noArg
  final def unary_-% (): SInt = macro SourceInfoTransform.noArg

  def unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = 0.S - this
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

  final def * (that: UInt): SInt = macro SourceInfoTransform.thatArg
  def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = {
    val thatToSInt = that.zext()
    val result = binop(sourceInfo, SInt(this.width + thatToSInt.width), TimesOp, thatToSInt)
    result.tail(1).asSInt
  }

  /** add (width +1) operator */
  final def +& (that: SInt): SInt = macro SourceInfoTransform.thatArg
  /** add (no growth) operator */
  final def +% (that: SInt): SInt = macro SourceInfoTransform.thatArg
  /** subtract (width +1) operator */
  final def -& (that: SInt): SInt = macro SourceInfoTransform.thatArg
  /** subtract (no growth) operator */
  final def -% (that: SInt): SInt = macro SourceInfoTransform.thatArg

  def do_+& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), AddOp, that)
  def do_+% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    (this +& that).tail(1).asSInt
  def do_-& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that)
  def do_-% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    (this -& that).tail(1).asSInt

  final def & (that: SInt): SInt = macro SourceInfoTransform.thatArg
  final def | (that: SInt): SInt = macro SourceInfoTransform.thatArg
  final def ^ (that: SInt): SInt = macro SourceInfoTransform.thatArg

  def do_& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that).asSInt
  def do_| (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that).asSInt
  def do_^ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that).asSInt

  /** Returns this wire bitwise-inverted. */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    unop(sourceInfo, UInt(width = width), BitNotOp).asSInt

  override def do_< (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
  final def != (that: SInt): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: SInt): Bool = macro SourceInfoTransform.thatArg
  final def === (that: SInt): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)

//  final def abs(): UInt = macro SourceInfoTransform.noArg

  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = {
    Mux(this < 0.S, (-this), this)
  }

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width + that), ShiftLeftOp, that)
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width.shiftRight(that)), ShiftRightOp, that)
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    this >> that.toInt
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
  
  override def do_asInterval(range: IntervalRange = IntervalRange.unknownRange)
                            (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {                       
    (range.lower, range.upper, range.binaryPoint) match {
      case (lx: IsKnown, ux: IsKnown, KnownBinaryPoint(bp)) =>
        // No mechanism to pass open/close to firrtl so need to handle directly
        // TODO: (chick) can we pass open/close to firrtl?
        val l = lx match {
          case Open(x) => x + math.pow(2, -bp)
          case Closed(x) => x
        }
        val u = ux match { 
          case Open(x) => x - math.pow(2, -bp)
          case Closed(x) => x
        }
        //TODO: (chick) Need to determine, what asInterval needs, and why it might need min and max as args -- CAN IT BE UNKNOWN?
        // Angie's operation: Decimal -> Int -> Decimal loses information. Need to be conservative here?
        val minBI = (l * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        val maxBI = (u * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.CEILING).toBigIntExact.get
        pushOp(DefPrim(sourceInfo, Interval(width, range), AsIntervalOp, ref, ILit(minBI), ILit(maxBI), ILit(bp)))
      case _ =>
        throwException(
          s"cannot call $this.asInterval($range), you must specify a known binaryPoint and range")
    }
  }

  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
    this := that.asSInt
  }
}

trait SIntFactory {
  /** Create an SInt type with inferred width. */
  def apply(): SInt = apply(Width())
  /** Create a SInt type or port with fixed width. */
  def apply(width: Width): SInt = new SInt(width)

  /** Create a SInt with the specified range */
  def apply(range: IntervalRange): SInt = {
    apply(range.getWidth)
  }
//TODO: fix this later
//  /** Create a SInt with the specified range */
//  def apply(range: (NumericBound[Int], NumericBound[Int])): SInt = {
//    apply(KnownSIntRange(range._1, range._2))
//  }

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
sealed class Bool(lit: Option[ULit] = None) extends UInt(1.W, lit) {
  private[core] override def cloneTypeWidth(w: Width): this.type = {
    require(!w.known || w.get == 1)
    new Bool().asInstanceOf[this.type]
  }

  // REVIEW TODO: Why does this need to exist and have different conventions
  // than Bits?
  final def & (that: Bool): Bool = macro SourceInfoTransform.thatArg
  final def | (that: Bool): Bool = macro SourceInfoTransform.thatArg
  final def ^ (that: Bool): Bool = macro SourceInfoTransform.thatArg

  def do_& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    binop(sourceInfo, Bool(), BitAndOp, that)
  def do_| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    binop(sourceInfo, Bool(), BitOrOp, that)
  def do_^ (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    binop(sourceInfo, Bool(), BitXorOp, that)

  /** Returns this wire bitwise-inverted. */
  override def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    unop(sourceInfo, Bool(), BitNotOp)

  /** Outputs the logical OR of two Bools.
   */
  def || (that: Bool): Bool = macro SourceInfoTransform.thatArg

  def do_|| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this | that

  /** Outputs the logical AND of two Bools.
   */
  def && (that: Bool): Bool = macro SourceInfoTransform.thatArg

  def do_&& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this & that

  /** Reinterprets this Bool as a Clock.  */
  def asClock(): Clock = macro SourceInfoTransform.noArg

  def do_asClock(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Clock = pushOp(DefPrim(sourceInfo, Clock(), AsClockOp, ref))
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
  private[core] override def typeEquivalent(that: Data): Boolean = that match {
    case that: FixedPoint => this.width == that.width && this.binaryPoint == that.binaryPoint  // TODO: should this be true for unspecified widths?
    case _ => false
  }

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new FixedPoint(w, binaryPoint).asInstanceOf[this.type]
  private[chisel3] def toType = s"Fixed$width$binaryPoint"

  override def connect (that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: FixedPoint => super.connect(that)
    case _ => this badConnect that
  }

  final def unary_- (): FixedPoint = macro SourceInfoTransform.noArg
  final def unary_-% (): FixedPoint = macro SourceInfoTransform.noArg

  def unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = FixedPoint.fromBigInt(0) - this
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

  final def * (that: UInt): FixedPoint = macro SourceInfoTransform.thatArg
  def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that.width, binaryPoint), TimesOp, that)

  final def * (that: SInt): FixedPoint = macro SourceInfoTransform.thatArg
  def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that.width, binaryPoint), TimesOp, that)

  /** add (width +1) operator */
  final def +& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  /** add (no growth) operator */
  final def +% (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  /** subtract (width +1) operator */
  final def -& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  /** subtract (no growth) operator */
  final def -% (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  def do_+& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint((this.width max that.width) + 1, this.binaryPoint max that.binaryPoint), AddOp, that)
  def do_+% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    (this +& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)
  def do_-& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint((this.width max that.width) + 1, this.binaryPoint max that.binaryPoint), SubOp, that)
  def do_-% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    (this -& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)

  final def & (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  final def | (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg
  final def ^ (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  def do_& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    throwException(s"And is illegal between $this and $that")
  def do_| (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    throwException(s"Or is illegal between $this and $that")
  def do_^ (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    throwException(s"Xor is illegal between $this and $that")

  final def setBinaryPoint(that: Int): FixedPoint = macro SourceInfoTransform.thatArg

  def do_setBinaryPoint(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = this.binaryPoint match {
    case KnownBinaryPoint(value) =>
      binop(sourceInfo, FixedPoint(this.width + (that - value), KnownBinaryPoint(that)), SetBinaryPoint, that)
    case _ =>
      binop(sourceInfo, FixedPoint(UnknownWidth(), KnownBinaryPoint(that)), SetBinaryPoint, that)
  }

  /** Returns this wire bitwise-inverted. */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    throwException(s"Not is illegal on $this")

  // TODO(chick): Consider comparison with UInt and SInt
  override def do_< (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  final def != (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg
  final def =/= (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg
  final def === (that: FixedPoint): Bool = macro SourceInfoTransform.thatArg

  def do_!= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=/= (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)

  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    // TODO: remove this once we have CompileOptions threaded through the macro system.
    import chisel3.core.ExplicitCompileOptions.NotStrict
    Mux(this < 0.F(0.BP), 0.F(0.BP) - this, this)
  }

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that, this.binaryPoint), ShiftLeftOp, that)
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    (this << that.toInt).asFixedPoint(this.binaryPoint)
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width.dynamicShiftLeft(that.width), this.binaryPoint), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width.shiftRight(that), this.binaryPoint), ShiftRightOp, that)
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    (this >> that.toInt).asFixedPoint(this.binaryPoint)
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

  // TODO: (chick) -- this is an invalid PrimOp (not enough constant args)
  def do_asInterval(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        //val iLit = ILit(value)
        //pushOp(DefPrim(sourceInfo, Interval(width, binaryPoint), AsIntervalOp, ref, iLit))
        throwException(s"INVALID asInterval")
      case _ =>
        throwException(s"cannot call $this.asInterval(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
    }
  }

  override def do_asInterval(range: IntervalRange = IntervalRange.unknownRange)
                            (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {                       
    (range.lower, range.upper, range.binaryPoint) match {
      case (lx: IsKnown, ux: IsKnown, KnownBinaryPoint(bp)) =>
        // No mechanism to pass open/close to firrtl so need to handle directly
        // TODO: (chick) can we pass open/close to firrtl?
        val l = lx match {
          case Open(x) => x + math.pow(2, -bp)
          case Closed(x) => x
        }
        val u = ux match { 
          case Open(x) => x - math.pow(2, -bp)
          case Closed(x) => x
        }
        //TODO: (chick) Need to determine, what asInterval needs, and why it might need min and max as args -- CAN IT BE UNKNOWN?
        // Angie's operation: Decimal -> Int -> Decimal loses information. Need to be conservative here?
        val minBI = (l * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.FLOOR).toBigIntExact.get
        val maxBI = (u * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.CEILING).toBigIntExact.get
        pushOp(DefPrim(sourceInfo, Interval(width, range), AsIntervalOp, ref, ILit(minBI), ILit(maxBI), ILit(bp)))
      case _ =>
        throwException(
          s"cannot call $this.asInterval($range), you must specify a known binaryPoint and range")
    }
  }

  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
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
  /** Create an FixedPoint type with inferred width. */
  def apply(): FixedPoint = apply(Width(), BinaryPoint())

  /** Create an FixedPoint type or port with fixed width. */
  @deprecated("Use FixedPoint(width: Width, binaryPoint: BinaryPoint) example FixedPoint(16.W, 8.BP)", "chisel3")
  def apply(width: Int, binaryPoint: Int): FixedPoint = apply(Width(width), BinaryPoint(binaryPoint))

  /** Create an FixedPoint type or port with fixed width. */
  def apply(width: Width, binaryPoint: BinaryPoint): FixedPoint = new FixedPoint(width, binaryPoint)

  /** Create an FixedPoint port with inferred width. */
  def apply(dir: Direction): FixedPoint = apply(dir, Width(), BinaryPoint())

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
  @deprecated("use fromDouble(value: Double, width: Width, binaryPoint: BinaryPoint)", "chisel3")
  def fromDouble(value: Double, dummy: PrivateType = PrivateObject,
                 width: Int = -1, binaryPoint: Int = 0): FixedPoint = {
    fromBigInt(
      toBigInt(value, binaryPoint), width = width, binaryPoint = binaryPoint
    )
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
    val multiplier = math.pow(2,binaryPoint)
    val result = i.toDouble / multiplier
    result
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
  * @param range       a range specifies min, max and binary point
  * @param lit
  */
sealed class Interval private[core] (
                                      width: Width,
                                      val range: chisel3.internal.firrtl.IntervalRange,
                                      lit: Option[IntervalLit] = None)
  extends Bits(width, lit) with Num[Interval] {
  private[core] override def cloneTypeWidth(w: Width): this.type =
    new Interval(w, range).asInstanceOf[this.type]

  //  private[chisel3] def toType = s"Interval$width$binaryPoint"

  private[chisel3] def toType = s"${range.serialize}"

  private[core] override def typeEquivalent(that: Data): Boolean =
    that.isInstanceOf[Interval] && this.width == that.width

  def binaryPoint: BinaryPoint = range.binaryPoint

  override def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: Interval => super.connect(that)
    case _ => this badConnect that
  }

  final def unary_-(): Interval = macro SourceInfoTransform.noArg
  final def unary_-%(): Interval = macro SourceInfoTransform.noArg

  def unary_-(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = Interval.fromBigInt(0) - this
  def unary_-%(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = Interval.fromBigInt(0) -% this

  /** add (default - growing) operator */
  override def do_+(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    this +& that
  /** subtract (default - growing) operator */
  override def do_-(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    this -& that
  override def do_*(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    binop(sourceInfo, Interval(this.width + that.width, this.range * that.range), TimesOp, that)

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

  def do_+&(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    binop(
      sourceInfo,
      Interval((this.width max that.width) + 1, this.range +& that.range),
      AddOp, that)
  def do_+%(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = throwException(s"Non-growing addition is not supported on Intervals")
  def do_-&(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    binop(sourceInfo,
      Interval((this.width max that.width) + 1, this.range -& that.range),
      SubOp, that)
  def do_-%(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = throwException(s"Non-growing subtraction is not supported on Intervals")

  final def &(that: Interval): Interval = macro SourceInfoTransform.thatArg
  final def |(that: Interval): Interval = macro SourceInfoTransform.thatArg
  final def ^(that: Interval): Interval = macro SourceInfoTransform.thatArg

  def do_&(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    throwException(s"And is illegal between $this and $that")
  def do_|(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    throwException(s"Or is illegal between $this and $that")
  def do_^(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    throwException(s"Xor is illegal between $this and $that")

  final def setBinaryPoint(that: Int): Interval = macro SourceInfoTransform.thatArg

  // Precision change changes range -- see firrtl PrimOps (requires floor)
  // aaa.bbb -> aaa.bb for sbp(2)
  // TODO: (chick) Match firrtl constraints
  def do_setBinaryPoint(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    binop(sourceInfo,
      Interval(UnknownWidth(),
        new IntervalRange(
          UnknownBound, UnknownBound, IntervalRange.getBinaryPoint(that))), SetBinaryPoint, that)

  final def shiftLeftBinaryPoint(that: Int): Interval = macro SourceInfoTransform.thatArg
  // aaa.bbb -> aaa.bbb00 (bpsl(2)): increase precision
  def do_shiftLeftBinaryPoint(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    val newBinaryPoint = this.range.binaryPoint + BinaryPoint(that)
    val newIntervalRange = IntervalRange(this.range.lower, this.range.upper, newBinaryPoint)
    binop(sourceInfo, Interval(this.width + that, newIntervalRange), ShiftLeftBinaryPoint, that)
  }

  final def shiftRightBinaryPoint(that: Int): Interval = macro SourceInfoTransform.thatArg
  // aaa.bbb -> aaa.b (bpsr(2)): decrease precision
  def do_shiftRightBinaryPoint(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    val newBinaryPoint = this.range.binaryPoint + BinaryPoint(-that)
    // TODO: (chick) Match firrtl constraints -- decreasing precision changes ranges
    val newIntervalRange = IntervalRange(UnknownBound, UnknownBound, newBinaryPoint)
    // TODO: (chick) How to subtract width?
    //binop(sourceInfo, Interval(this.width + KnownWidth(-that), newIntervalRange), ShiftRightBinaryPoint, that)
    binop(sourceInfo, Interval(UnknownWidth(), newIntervalRange), ShiftRightBinaryPoint, that)
  }

  /** Returns this wire bitwise-inverted. */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    throwException(s"Not is illegal on $this")

  // TODO(chick): Consider comparison with UInt and SInt
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
    Mux(this < Interval.fromBigInt(0), (Interval.fromBigInt(0) - this), this)
  }

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    binop(sourceInfo, Interval(this.width + that, this.range << that), ShiftLeftOp, that)
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    do_<<(that.toInt)
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    val newRange = that.width match {
      // TODO: (chick) should be this.range << (2^(w) - 1); leave firrtl to figure this out for now
      //case w: KnownWidth => this.range << w
      case _ => IntervalRange(UnknownBound, UnknownBound, this.range.binaryPoint)
    }
    binop(sourceInfo,
      Interval(this.width.dynamicShiftLeft(that.width), newRange),
      DynamicShiftLeftOp, that)
  }
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    binop(sourceInfo,
      Interval(this.width.shiftRight(that), this.range >> that), ShiftRightOp, that)
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval =
    do_>>(that.toInt)
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    // Worst case range is when you shift by nothing
    val newRange = this.range
    binop(sourceInfo, Interval(this.width, newRange), DynamicShiftRightOp, that)
  }

  final def wrap(that: Interval): Interval = macro SourceInfoTransform.thatArg
  // TODO: (chick) port correct firrtl constraints (requires 2^x)
  def do_wrap(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    val dest = Interval(IntervalRange(UnknownBound, UnknownBound, this.range.binaryPoint))
    val other = that
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(other, s"'other' ($other)")
    pushOp(DefPrim(sourceInfo, dest, WrapOp, this.ref, other.ref, ILit(0)))
  }

  // Reassign interval without actually adding any logic
  final def reassignInterval(that: Interval): Interval = macro SourceInfoTransform.thatArg
  def do_reassignInterval(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    val dest = Interval(IntervalRange(UnknownBound, UnknownBound, this.range.binaryPoint))
    val other = that
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(other, s"'other' ($other)")
    pushOp(DefPrim(sourceInfo, dest, WrapOp, this.ref, other.ref, ILit(1)))
  }

  final def wrap(that: UInt): Interval = macro SourceInfoTransform.thatArg
  def do_wrap(that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    //binop(sourceInfo, TypePropagate(_root_.firrtl.PrimOps.Wrap, Seq(this, that), Nil), WrapOp, that)
    require(that.widthKnown, "UInt clip width must be known")
    val u = (1 << that.getWidth) - 1
    do_wrap(Wire(Interval(IntervalRange(Closed(0), Closed(u), BinaryPoint(0)))))
    // TODO: (chick) Can this be done w/o known width?
  }

  final def wrap(that: SInt): Interval = macro SourceInfoTransform.thatArg
  def do_wrap(that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    //binop(sourceInfo, TypePropagate(_root_.firrtl.PrimOps.Wrap, Seq(this, that), Nil), WrapOp, that)
    // TODO: (chick) same as above
    require(that.widthKnown, "SInt clip width must be known")
    val l = -(1 << (that.getWidth - 1))
    val u = (1 << (that.getWidth - 1)) - 1
    do_wrap(Wire(Interval(IntervalRange(Closed(l), Closed(u), BinaryPoint(0)))))
  }

  final def wrap(that: IntervalRange): Interval = macro SourceInfoTransform.thatArg
  def do_wrap(that: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    do_wrap(Wire(Interval(that)))
  }

  final def clip(that: Interval): Interval = macro SourceInfoTransform.thatArg
  def do_clip(that: Interval)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    binop(sourceInfo, Interval(IntervalRange(UnknownBound, UnknownBound, this.range.binaryPoint)), ClipOp, that)
  }

  final def clip(that: UInt): Interval = macro SourceInfoTransform.thatArg
  def do_clip(that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    require(that.widthKnown, "UInt clip width must be known")
    val u = (1 << that.getWidth) - 1
    do_clip(Wire(Interval(IntervalRange(Closed(0), Closed(u), BinaryPoint(0)))))
    // TODO: (chick) Can this be done w/o known width?
    //binop(sourceInfo, TypePropagate(_root_.firrtl.PrimOps.Clip, Seq(this, that), Nil), ClipOp, that)
  }

  final def clip(that: SInt): Interval = macro SourceInfoTransform.thatArg
  def do_clip(that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    // TODO: (chick) same as above
    require(that.widthKnown, "SInt clip width must be known")
    val l = -(1 << (that.getWidth - 1))
    val u = (1 << (that.getWidth - 1)) - 1
    do_clip(Wire(Interval(IntervalRange(Closed(l), Closed(u), BinaryPoint(0)))))
    //binop(sourceInfo, TypePropagate(_root_.firrtl.PrimOps.Clip, Seq(this, that), Nil), ClipOp, that)
  }

  final def clip(that: IntervalRange): Interval = macro SourceInfoTransform.thatArg
  def do_clip(that: IntervalRange)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    do_clip(Wire(Interval(that)))
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
        // TODO: (chick) Why is there both ILit and IntervalLit (?) -- seems like there's some butchering of notation?
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(
          s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
    }
  }

  // TODO: (chick) INVALID -- not enough args
  def do_asInterval(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Interval = {
    pushOp(DefPrim(sourceInfo, Interval(this.width, this.range), AsIntervalOp, ref))
    throwException("asInterval INVALID")
  }

  // TODO: (chick) looks like this is wrong and only for FP?
  def do_fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = {
    /*val res = Wire(this, null).asInstanceOf[this.type]
    res := (that match {
      case fp: FixedPoint => fp.asSInt.asFixedPoint(this.binaryPoint)
      case _ => that.asFixedPoint(this.binaryPoint)
    })
    res*/
    throwException("fromBits INVALID for intervals")
  }

  private[core] override def connectFromBits(that: Bits)
      (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
    this := that.asInterval(this.binaryPoint, this.range)
  }
  //TODO(chick): Consider "convert" as an arithmetic conversion to UInt/SInt
}

trait IntervalFactory {
  // TODO: (chick) -- formalize
  /** Create a Interval type with inferred width and binary point. */
  def apply(): Interval = apply(Width(), UnknownBinaryPoint)
  /** Create a Interval type with specified width. */
  def apply(width: Width): Interval = Interval(width, 0.BP)
  /** Create a Interval type with specified width. */
  def apply(width: Width, binaryPoint: BinaryPoint): Interval = {
    Interval(width, new IntervalRange(UnknownBound, UnknownBound, IntervalRange.getBinaryPoint(binaryPoint)))
  }
  
  // TODO: (chick) Might want to round the range for asInterval as well?
  /** Create a Interval type with specified width. */
  def apply(width: Width, range: chisel3.internal.firrtl.IntervalRange): Interval = {
    (range.lower, range.upper, range.binaryPoint) match {
      case (lx: Bound, ux: Bound, KnownBinaryPoint(bp)) =>
        // No mechanism to pass open/close to firrtl so need to handle directly -> convert to Closed
        // TODO: (chick) can we pass open/close to firrtl?
        val lower = lx match {
          case Open(x) => 
            val l = x + math.pow(2, -bp)
            val min = (l * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.FLOOR) / BigDecimal(1 << bp)
            Closed(min)
          case Closed(x) => 
            val l = x
            val min = (l * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.FLOOR) / BigDecimal(1 << bp)
            Closed(min)
          case _ => 
            lx
        }
        val upper = ux match {
          case Open(x) => 
            val u = x - math.pow(2, -bp)
            val max = (u * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.CEILING) / BigDecimal(1 << bp)
            Closed(max)
          case Closed(x) => 
            val u = x
            val max = (u * math.pow(2, bp)).setScale(0, BigDecimal.RoundingMode.CEILING) / BigDecimal(1 << bp)
            Closed(max)
          case _ => 
            ux
        }
        val newRange = chisel3.internal.firrtl.IntervalRange(lower, upper, range.binaryPoint)
        new Interval(width, newRange)
      case _ =>
        new Interval(width, range)
    }  
  }

  /** Create a Interval literal with specified width. -- Note: Double already converted to BigInt via multiplication */
  protected[chisel3] def Lit(value: BigInt, width: Width, binaryPoint: BinaryPoint): Interval = {
    // TODO: (chick) Fundamental problem with the way IntervalLits are implemented in Chisel -- 
    // You need to perform an "asInterval" at some point, which requires conversion from double to
    // BigInt representation
    binaryPoint match {
      case KnownBinaryPoint(bp) =>
      case _ => throw new Exception("Lit bp must be known!")
    }
    val lit = IntervalLit(value, width, binaryPoint)
    val result = new Interval(
      width,
      new IntervalRange(
        Closed(BigDecimal(value) / BigDecimal(1 << binaryPoint.get)),
        Closed(BigDecimal(value) / BigDecimal(1 << binaryPoint.get)),
        IntervalRange.getBinaryPoint(binaryPoint)
      ), Some(lit))
    // Bind result to being an Literal
    result.binding = LitBinding()
    result
  }

  /** Create a Interval with the specified range */
  def apply(range: IntervalRange): Interval = {
    val result = apply(range.getWidth, range)
    result
  }

  def fromBigInt(value: BigInt, width: Int = -1, binaryPoint: Int = 0): Interval =
    // Passed to Firrtl as BigInt
    if(width == -1) {
      apply(value, Width(), BinaryPoint(binaryPoint))
    }
    else {
      apply(value, Width(width), BinaryPoint(binaryPoint))
    }
  /** Create an Interval literal with inferred width from Double.
    * Use PrivateObject to force users to specify width and binaryPoint by name
    */
  def fromDouble(value: Double, dummy: PrivateType = PrivateObject,
                 width: Int = -1, binaryPoint: Int = 0): Interval = {
    fromBigInt(
      toBigInt(value, binaryPoint), width = width, binaryPoint = binaryPoint
    )
  }

  def apply(value: BigInt, width: Width, binaryPoint: BinaryPoint): Interval = {
    // TODO: (chick) Fundamental problem with the way IntervalLits are implemented in Chisel -- 
    // You need to perform an "asInterval" at some point, which requires conversion from double to
    // BigInt representation
    binaryPoint match {
      case KnownBinaryPoint(bp) =>
      case _ => throw new Exception("Lit bp must be known!")
    }
    // Double already converted to BigInt by multiplying up
    val lit = IntervalLit(value, width, binaryPoint)
    val range = IntervalRange(
        Closed(BigDecimal(value) / BigDecimal(1 << binaryPoint.get)),
        Closed(BigDecimal(value) / BigDecimal(1 << binaryPoint.get)),
        IntervalRange.getBinaryPoint(binaryPoint)
    )
    val newLiteral = new Interval(lit.width, range, Some(lit))
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
/**
  * Factory and convenience methods for the Interval class
  * IMPORTANT: The API provided here is experimental and may change in the future.
  */
object Interval extends IntervalFactory

/** Data type for representing bidirectional bitvectors of a given width
  *
  * Analog support is limited to allowing wiring up of Verilog BlackBoxes with bidirectional (inout)
  * pins. There is currently no support for reading or writing of Analog types within Chisel code.
  *
  * Given that Analog is bidirectional, it is illegal to assign a direction to any Analog type. It
  * is legal to "flip" the direction (since Analog can be a member of aggregate types) which has no
  * effect.
  *
  * Analog types are generally connected using the bidirectional [[attach]] mechanism, but also
  * support limited bulkconnect `<>`. Analog types are only allowed to be bulk connected *once* in a
  * given module. This is to prevent any surprising consequences of last connect semantics.
  *
  * @note This API is experimental and subject to change
  */
final class Analog private (width: Width) extends Element(width) {
  require(width.known, "Since Analog is only for use in BlackBoxes, width must be known")

  private[chisel3] def toType = s"Analog$width"

  private[core] override def typeEquivalent(that: Data): Boolean =
    that.isInstanceOf[Analog] && this.width == that.width

  def cloneType: this.type = new Analog(width).asInstanceOf[this.type]

  // Used to enforce single bulk connect of Analog types, multi-attach is still okay
  // Note that this really means 1 bulk connect per Module because a port can
  //   be connected in the parent module as well
  private[core] val biConnectLocs = mutable.Map.empty[UserModule, SourceInfo]

  // Define setter/getter pairing
  // Analog can only be bound to Ports and Wires (and Unbound)
  private[core] override def binding_=(target: Binding): Unit = target match {
    case (_: UnboundBinding | _: WireBinding | PortBinding(_, None)) => super.binding_=(target)
    case _ => throwException("Only Wires and Ports can be of type Analog")
  }

  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    throwException("Analog does not support asUInt")

  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
                                                         compileOptions: CompileOptions): Unit = {
    throwException("Analog does not support connectFromBits")
  }

  final def toPrintable: Printable = PString("Analog")
}
/** Object that provides factory methods for [[Analog]] objects
  *
  * @note This API is experimental and subject to change
  */
object Analog {
  def apply(width: Width): Analog = new Analog(width)
}
