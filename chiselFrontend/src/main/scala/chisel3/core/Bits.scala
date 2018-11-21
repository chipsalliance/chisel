// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros
import collection.mutable

import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, DeprecatedSourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform,
  UIntTransform}
import chisel3.internal.firrtl.PrimOp._

//scalastyle:off method.name

/** Element is a leaf data type: it cannot contain other [[Data]] objects. Example uses are for representing primitive
  * data types, like integers and bits.
  *
  * @define coll element
  */
abstract class Element extends Data {
  private[chisel3] final def allElements: Seq[Element] = Seq(this)
  def widthKnown: Boolean = width.known
  def name: String = getRef.name

  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection) {
    binding = target
    val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
    direction = resolvedDirection match {
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip => ActualDirection.Unspecified
      case SpecifiedDirection.Output => ActualDirection.Output
      case SpecifiedDirection.Input => ActualDirection.Input
    }
  }

  private[core] override def topBindingOpt: Option[TopBinding] = super.topBindingOpt match {
    // Translate Bundle lit bindings to Element lit bindings
    case Some(BundleLitBinding(litMap)) => litMap.get(this) match {
      case Some(litArg) => Some(ElementLitBinding(litArg))
      case _ => Some(DontCareBinding())
    }
    case topBindingOpt => topBindingOpt
  }

  private[core] def litArgOption: Option[LitArg] = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => Some(litArg)
    case _ => None
  }

  override def litOption: Option[BigInt] = litArgOption.map(_.num)
  private[core] def litIsForcedWidth: Option[Boolean] = litArgOption.map(_.forcedWidth)

  // provide bits-specific literal handling functionality here
  override private[chisel3] def ref: Arg = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => litArg
    case Some(BundleLitBinding(litMap)) => litMap.get(this) match {
      case Some(litArg) => litArg
      case _ => throwException(s"internal error: DontCare should be caught before getting ref")
    }
    case _ => super.ref
  }

  private[core] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit = {
    // If the source is a DontCare, generate a DefInvalid for the sink,
    //  otherwise, issue a Connect.
    if (that == DontCare) {
      pushCommand(DefInvalid(sourceInfo, Node(this)))
    } else {
      pushCommand(Connect(sourceInfo, Node(this), that.ref))
    }
  }
}

/** Exists to unify common interfaces of [[Bits]] and [[Reset]].
  *
  * @note This is a workaround because macros cannot override abstract methods.
  */
private[chisel3] sealed trait ToBoolable extends Element {

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
  private[core] def cloneTypeWidth(width: Width): this.type

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
      (((value >> x.toInt) & 1) == 1).asBool
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
    apply(x.toInt, y.toInt)

  private[core] def unop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp): T = {
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref))
  }
  private[core] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: BigInt): T = {
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, ILit(other)))
  }
  private[core] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: Bits): T = {
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, other.ref))
  }
  private[core] def compop(sourceInfo: SourceInfo, op: PrimOp, other: Bits): Bool = {
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }
  private[core] def redop(sourceInfo: SourceInfo, op: PrimOp): Bool = {
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
    * @note The width of the returned $coll is `width of this + pow(2, width of that)`.
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

  /** Reinterpret this $coll as a [[SInt]]
    *
    * @note The value is not guaranteed to be preserved. For example, a [[UInt]] of width 3 and value 7 (0b111) would
    * become a [[SInt]] with value -1.
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

  /** Reinterpret cast to Bits. */
  @chiselRuntimeDeprecated
  @deprecated("Use asUInt, which does the same thing but returns a more concrete type", "chisel3")
  final def asBits(implicit compileOptions: CompileOptions): Bits = {
    implicit val sourceInfo = DeprecatedSourceInfo
    do_asUInt
  }

  @chiselRuntimeDeprecated
  @deprecated("Use asSInt, which makes the reinterpret cast more explicit", "chisel3")
  final def toSInt(implicit compileOptions: CompileOptions): SInt = {
    implicit val sourceInfo = DeprecatedSourceInfo
    do_asSInt
  }

  @chiselRuntimeDeprecated
  @deprecated("Use asUInt, which makes the reinterpret cast more explicit", "chisel3")
  final def toUInt(implicit compileOptions: CompileOptions): UInt = {
    implicit val sourceInfo = DeprecatedSourceInfo
    do_asUInt
  }

  final def do_toBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = {
    width match {
      case KnownWidth(1) => this(0)
      case _ => throwException(s"can't covert UInt<$width> to Bool")
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

// REVIEW TODO: Further discussion needed on what Num actually is.
/** Abstract trait defining operations available on numeric-like hardware data types.
  *
  * @tparam T the underlying type of the number
  * @groupdesc Arithmetic Arithmetic hardware operators
  * @groupdesc Comparison Comparison hardware operators
  * @groupdesc Logical Logical hardware operators
  * @define coll numeric-like type
  * @define numType hardware type
  * @define canHaveHighCost can result in significant cycle time and area costs
  * @define canGenerateA This method can generate a
  * @define singleCycleMul  @note $canGenerateA single-cycle multiplier which $canHaveHighCost.
  * @define singleCycleDiv  @note $canGenerateA single-cycle divider which $canHaveHighCost.
  * @define maxWidth        @note The width of the returned $numType is `max(width of this, width of that)`.
  * @define maxWidthPlusOne @note The width of the returned $numType is `max(width of this, width of that) + 1`.
  * @define sumWidth        @note The width of the returned $numType is `width of this` + `width of that`.
  * @define unchangedWidth  @note The width of the returned $numType is unchanged, i.e., the `width of this`.
  */
abstract trait Num[T <: Data] {
  self: Num[T] =>
  // def << (b: T): T
  // def >> (b: T): T
  //def unary_-(): T

  // REVIEW TODO: double check ops conventions against FIRRTL

  /** Addition operator
    *
    * @param that a $numType
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def + (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Multiplication operator
    *
    * @param that a $numType
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def * (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_* (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Division operator
    *
    * @param that a $numType
    * @return the quotient of this $coll divided by `that`
    * $singleCycleDiv
    * @todo full rules
    * @group Arithmetic
    */
  final def / (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_/ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Modulo operator
    *
    * @param that a $numType
    * @return the remainder of this $coll divided by `that`
    * $singleCycleDiv
    * @group Arithmetic
    */
  final def % (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_% (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Subtraction operator
    *
    * @param that a $numType
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def - (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_- (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Less than operator
    *
    * @param that a $numType
    * @return a hardware [[Bool]] asserted if this $coll is less than `that`
    * @group Comparison
    */
  final def < (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Less than or equal to operator
    *
    * @param that a $numType
    * @return a hardware [[Bool]] asserted if this $coll is less than or equal to `that`
    * @group Comparison
    */
  final def <= (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Greater than operator
    *
    * @param that a hardware component
    * @return a hardware [[Bool]] asserted if this $coll is greater than `that`
    * @group Comparison
    */
  final def > (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Greater than or equal to operator
    *
    * @param that a hardware component
    * @return a hardware [[Bool]] asserted if this $coll is greather than or equal to `that`
    * @group Comparison
    */
  final def >= (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Absolute value operator
    *
    * @return a $numType with a value equal to the absolute value of this $coll
    * $unchangedWidth
    * @group Arithmetic
    */
  final def abs(): T = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Minimum operator
    *
    * @param that a hardware $coll
    * @return a $numType with a value equal to the mimimum value of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def min(that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_min(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    Mux(this < that, this.asInstanceOf[T], that)

  /** Maximum operator
    *
    * @param that a $numType
    * @return a $numType with a value equal to the mimimum value of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def max(that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_max(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    Mux(this < that, that, this.asInstanceOf[T])
}

/** A data type for unsigned integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[UInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class UInt private[core] (width: Width) extends Bits(width) with Num[UInt] {

  private[core] override def typeEquivalent(that: Data): Boolean =
    that.isInstanceOf[UInt] && this.width == that.width

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new UInt(w).asInstanceOf[this.type]

  // TODO: refactor to share documentation with Num or add independent scaladoc
  /** Unary negation (expanding width)
    *
    * @return a $coll equal to zero minus this $coll
    * $expandingWidth
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
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that).asUInt
  /** @group SourceInfoTransformMacro */
  def do_-% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this -& that).tail(1)

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
  def do_andR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = ~this === 0.U
  /** @group SourceInfoTransformMacro */
  def do_xorR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = redop(sourceInfo, XorReduceOp)

  override def do_< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)

  @chiselRuntimeDeprecated
  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
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
    * @return a hardware [[Bool]] asserted if the least significant bit of this $coll is zero
    * @group Bitwise
    */
  final def unary_! () : Bool = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_! (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : Bool = this === 0.U(1.W)

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this >> that.toInt
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
    val result = new UInt(lit.width)
    // Bind result to being an Literal
    lit.bindLitArg(result)
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

/** A data type for signed integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[SInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class SInt private[core] (width: Width) extends Bits(width) with Num[SInt] {

  private[core] override def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass && this.width == that.width  // TODO: should this be true for unspecified widths?

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new SInt(w).asInstanceOf[this.type]

  /** Unary negation (expanding width)
    *
    * @return a hardware $coll equal to zero minus this $coll
    * $expandingWidth
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
  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
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
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
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
    val result = new SInt(lit.width)
    lit.bindLitArg(result)
  }
}

object SInt extends SIntFactory

sealed trait Reset extends Element with ToBoolable

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  *
  * @define coll [[Bool]]
  * @define numType $coll
  */
sealed class Bool() extends UInt(1.W) with Reset {
  private[core] override def cloneTypeWidth(w: Width): this.type = {
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
    * @note this is equivalent to [[Bool.|]]
    * @group Logical
    */
  def || (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_|| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this | that

  /** Logical and operator
    *
    * @param that a hardware $coll
    * @return the lgocial and of this $coll and `that`
    * @note this is equivalent to [[Bool.&]]
    * @group Logical
    */
  def && (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this & that

  /** Reinterprets this $coll as a clock */
  def asClock(): Clock = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asClock(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Clock = pushOp(DefPrim(sourceInfo, Clock(), AsClockOp, ref))
}

trait BoolFactory {
  /** Creates an empty Bool.
   */
  def apply(): Bool = new Bool()

  /** Creates Bool literal.
   */
  protected[chisel3] def Lit(x: Boolean): Bool = {
    val result = new Bool()
    val lit = ULit(if (x) 1 else 0, Width(1))
    // Ensure we have something capable of generating a name.
    lit.bindLitArg(result)
  }
}

object Bool extends BoolFactory

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
  * @define coll [[FixedPoint]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class FixedPoint private (width: Width, val binaryPoint: BinaryPoint)
    extends Bits(width) with Num[FixedPoint] {
  private[core] override def typeEquivalent(that: Data): Boolean = that match {
    case that: FixedPoint => this.width == that.width && this.binaryPoint == that.binaryPoint  // TODO: should this be true for unspecified widths?
    case _ => false
  }

  private[core] override def cloneTypeWidth(w: Width): this.type =
    new FixedPoint(w, binaryPoint).asInstanceOf[this.type]

  override def connect (that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
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
    * $expandingWidth
    * @group Arithmetic
    */
  final def unary_- (): FixedPoint = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus `this` shifted right by one
    * $constantWidth
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
    * $sumWidth
    * $singleCycleMul
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
    * $sumWidth
    * $singleCycleMul
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
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def +& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  final def +% (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def -& (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that` shifted right by one
    * $maxWidth
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
    * $maxWidth
    * @group Bitwise
    */
  final def & (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def | (that: FixedPoint): FixedPoint = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
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
    import chisel3.core.ExplicitCompileOptions.NotStrict
    Mux(this < 0.F(0.BP), 0.F(0.BP) - this, this)
  }

  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width + that, this.binaryPoint), ShiftLeftOp, validateShiftAmount(that))
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    (this << that.toInt).asFixedPoint(this.binaryPoint)
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width.dynamicShiftLeft(that.width), this.binaryPoint), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
    binop(sourceInfo, FixedPoint(this.width.shiftRight(that), this.binaryPoint), ShiftRightOp, validateShiftAmount(that))
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
  @chiselRuntimeDeprecated
  @deprecated("Use FixedPoint(width: Width, binaryPoint: BinaryPoint) example FixedPoint(16.W, 8.BP)", "chisel3")
  def apply(width: Int, binaryPoint: Int): FixedPoint = apply(Width(width), BinaryPoint(binaryPoint))

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
  @chiselRuntimeDeprecated
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
  def apply(value: BigInt, width: Width, binaryPoint: BinaryPoint): FixedPoint = {
    val lit = FPLit(value, width, binaryPoint)
    val newLiteral = new FixedPoint(lit.width, lit.binaryPoint)
    // Ensure we have something capable of generating a name.
    lit.bindLitArg(newLiteral)
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
final class Analog private (private[chisel3] val width: Width) extends Element {
  require(width.known, "Since Analog is only for use in BlackBoxes, width must be known")

  private[core] override def typeEquivalent(that: Data): Boolean =
    that.isInstanceOf[Analog] && this.width == that.width

  override def litOption = None

  def cloneType: this.type = new Analog(width).asInstanceOf[this.type]

  // Used to enforce single bulk connect of Analog types, multi-attach is still okay
  // Note that this really means 1 bulk connect per Module because a port can
  //   be connected in the parent module as well
  private[core] val biConnectLocs = mutable.Map.empty[UserModule, SourceInfo]

  // Define setter/getter pairing
  // Analog can only be bound to Ports and Wires (and Unbound)
  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection) {
    SpecifiedDirection.fromParent(parentDirection, specifiedDirection) match {
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip =>
      case x => throwException(s"Analog may not have explicit direction, got '$x'")
    }
    val targetTopBinding = target match {
      case target: TopBinding => target
      case ChildBinding(parent) => parent.topBinding
    }

    // Analog counts as different directions based on binding context
    targetTopBinding match {
      case WireBinding(_) => direction = ActualDirection.Unspecified  // internal wire
      case PortBinding(_) => direction = ActualDirection.Bidirectional(ActualDirection.Default)
      case x => throwException(s"Analog can only be Ports and Wires, not '$x'")
    }
    binding = target
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
