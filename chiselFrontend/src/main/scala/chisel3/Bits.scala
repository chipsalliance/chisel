// See LICENSE for license details.
package chisel3

import scala.language.experimental.macros
import chisel3.internal._
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{DeprecatedSourceInfo, SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform,
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
//scalastyle:off number.of.methods
sealed class Bits(private[chisel3] val width: Width) extends ToBoolable {
  // TODO: perhaps make this concrete?
  // Arguments for: self-checking code (can't do arithmetic on bits)
  // Arguments against: generates down to a FIRRTL UInt anyways

  // Only used for in a few cases, hopefully to be removed
  private[chisel3] def cloneTypeWidth(width: Width): this.type =
    new UInt(width).asInstanceOf[this.type]

  def cloneType: this.type = cloneTypeWidth(width)

  private[chisel3] override def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass && this.width == that.width  // TODO: should this be true for unspecified widths?

  protected final def validateShiftAmount(x: Int): Int = {
    if (x < 0) {
      Builder.error(s"Negative shift amounts are illegal (got $x)")
    }
    x
  }

  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
                                                            compileOptions: CompileOptions): Unit = {
    this := that
  }

  // REVIEW TODO: double check ops conventions against FIRRTL

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

  // TODO: shouldn't they return Bits?
  /** @group SourceInfoTransformMacro */
  def do_tail(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val w = width match {
      case KnownWidth(x) =>
        require(x >= n, s"Can't tail($n) for width $x < $n")
        Width(x - n)
      case UnknownWidth() => Width()
    }
    binop(sourceInfo, UInt(width = w), TailOp, n).asUInt
  }

  /** @group SourceInfoTransformMacro */
  def do_head(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    width match {
      case KnownWidth(x) => require(x >= n, s"Can't head($n) for width $x < $n")
      case UnknownWidth() =>
    }
    binop(sourceInfo, UInt(Width(n)), HeadOp, n).asUInt
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
    val theBits = this.asUInt >> x
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
  def asSInt(): SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width), AsSIntOp, ref))

  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))

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


  /** Bitwise inversion operator
   *
   * @return this $coll with each bit inverted
   * @group Bitwise
   */
  def unary_~ (): Bits = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    unop(sourceInfo, UInt(width = width), BitNotOp)

  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or T?
  def << (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  def << (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** Dynamic left shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
   * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
   * @group Bitwise
   */
  def << (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  // REVIEW TODO: redundant
  def >> (that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  def >> (that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** Dynamic right shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
   * significant bits.
   * $unchangedWidth
   * @group Bitwise
   */
  def >> (that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  /** @group SourceInfoTransformMacro */
  def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    this << castToInt(that, "Shift amount")
  /** @group SourceInfoTransformMacro */
  def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  /** @group SourceInfoTransformMacro */
  def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    binop(sourceInfo, UInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  /** @group SourceInfoTransformMacro */
  def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    this >> castToInt(that, "Shift amount")
  /** @group SourceInfoTransformMacro */
  def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    binop(sourceInfo, UInt(this.width), DynamicShiftRightOp, that)

  /** Bitwise and operator
   *
   * @param that a hardware $coll
   * @return the bitwise and of  this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def & (that: Bits): Bits = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
   *
   * @param that a hardware $coll
   * @return the bitwise or of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def | (that: Bits): Bits = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
   *
   * @param that a hardware $coll
   * @return the bitwise xor of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def ^ (that: Bits): Bits = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_& (that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    binop(sourceInfo, Bits(), BitAndOp, that)
  /** @group SourceInfoTransformMacro */
  def do_| (that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    binop(sourceInfo, Bits(), BitOrOp, that)
  /** @group SourceInfoTransformMacro */
  def do_^ (that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bits =
    binop(sourceInfo, Bits(), BitXorOp, that)
  
}

/** A data type for unsigned integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[UInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class UInt private[chisel3] (override val width: Width) extends Bits(width) with Num[UInt] with NumBits[UInt] {
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

  /** @group SourceInfoTransformMacro */
  def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : UInt = 0.U - this
  /** @group SourceInfoTransformMacro */
  def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = 0.U -% this

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

  //  override def abs: UInt = macro SourceInfoTransform.noArg
  override def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this

  /** @group SourceInfoTransformMacro */
  def do_& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that)
  /** @group SourceInfoTransformMacro */
  def do_| (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that)
  /** @group SourceInfoTransformMacro */
  def do_^ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that)

  /** Bitwise inversion operator
   *
   * @return this $coll with each bit inverted
   * @group Bitwise
   */
  override def unary_~ (): UInt = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  override def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    unop(sourceInfo, UInt(width = width), BitNotOp)

  @deprecated("unary_! should only be used with type Bool. Use explicit equality with 0 (x === 0) for UInt values", "chisel3")
  def unary_! () : Bool = macro SourceInfoTransform.noArg
  /** @group SourceInfoTransformMacro */
  def do_unary_! (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : Bool = this === 0.U

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

  /** Bitwise and operator
   *
   * @param that a hardware $coll
   * @return the bitwise and of  this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def & (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
   *
   * @param that a hardware $coll
   * @return the bitwise or of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def | (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
   *
   * @param that a hardware $coll
   * @return the bitwise xor of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def ^ (that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_orR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= 0.U
  /** @group SourceInfoTransformMacro */
  def do_andR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = width match {
    // Generate a simpler expression if the width is known
    case KnownWidth(w) => this === ((BigInt(1) << w) - 1).U
    case UnknownWidth() =>  ~this === 0.U
  }
  /** @group SourceInfoTransformMacro */
  def do_xorR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    redop(sourceInfo, XorReduceOp)

  @chiselRuntimeDeprecated
  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
  final def != (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= that

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

  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this

  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that.asUInt
  }

  private def subtractAsSInt(that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that)
}

trait BitsFactory {
  /** Create a UInt type with inferred width. */
  def apply(): Bits = apply(Width())
  /** Create a UInt port with specified width. */
  def apply(width: Width): Bits = new Bits(width)

  /** Create a UInt literal with specified width. */
  protected[chisel3] def Lit(value: BigInt, width: Width): Bits = {
    val lit = ULit(value, width)
    val result = new Bits(lit.width)
    // Bind result to being an Literal
    lit.bindLitArg(result)
  }
}

/** A data type for signed integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[SInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class SInt private[chisel3] (width: Width) extends Bits(width) with Num[SInt] with NumBits[SInt]{
  override def toString: String = {
    val bindingString = litOption match {
      case Some(value) => s"($value)"
      case _ => bindingToString
    }
    s"SInt$width$bindingString"
  }

  private[chisel3] override def cloneTypeWidth(w: Width): this.type =
    new SInt(w).asInstanceOf[this.type]


  /** @group SourceInfoTransformMacro */
  override def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = 0.S - this
  /** @group SourceInfoTransformMacro */
  override def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = 0.S -% this

  override def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width + that.width), TimesOp, that)
  override def do_/ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width), DivideOp, that)
  override def do_% (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, SInt(this.width), RemOp, that)

  // TODO(kamy) What's so special about *? How about mixed signed/unsigned/FP /,%,+,-,<,>,>=,<=?
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

  /** @group SourceInfoTransformMacro */
  def do_& (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that).asSInt
  /** @group SourceInfoTransformMacro */
  def do_| (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that).asSInt
  /** @group SourceInfoTransformMacro */
  def do_^ (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that).asSInt

  /** Bitwise inversion operator
   *
   * @return this $coll with each bit inverted
   * @group Bitwise
   */
  override def unary_~ (): SInt = macro SourceInfoWhiteboxTransform.noArg
  /** @group SourceInfoTransformMacro */
  override def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    unop(sourceInfo, UInt(width = width), BitNotOp).asSInt


  @chiselRuntimeDeprecated
  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
  final def != (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= that

  override def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = {
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

  override def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = this

  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
    this := that.asSInt
  }

  /** Bitwise and operator
   *
   * @param that a hardware $coll
   * @return the bitwise and of  this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def & (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
   *
   * @param that a hardware $coll
   * @return the bitwise or of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def | (that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
   *
   * @param that a hardware $coll
   * @return the bitwise xor of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def ^ (that: SInt): SInt = macro SourceInfoTransform.thatArg
}

sealed trait Reset extends Element with ToBoolable

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  *
  * @define coll [[Bool]]
  * @define numType $coll
  */
sealed class Bool extends UInt(1.W) with Reset {
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
   * $maxWidth
   * @group Bitwise
   */
  final def & (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
   *
   * @param that a hardware $coll
   * @return the bitwise or of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  final def | (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
   *
   * @param that a hardware $coll
   * @return the bitwise xor of this $coll and `that`
   * $maxWidth
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

  /** Bitwise inversion operator
   *
   * @return this $coll with each bit inverted
   * @group Bitwise
   */
  override def unary_~ (): Bool = macro SourceInfoWhiteboxTransform.noArg
  /** @group SourceInfoTransformMacro */
  override def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    unop(sourceInfo, Bool(), BitNotOp)

  /** Unary not
   *
   * @return a hardware [[Bool]] asserted if this $coll equals zero
   * @group Bitwise
   */
  final override def unary_! () : Bool = macro SourceInfoTransform.noArg
  /** @group SourceInfoTransformMacro */
  override def do_unary_! (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : Bool = this === 0.U

  /** Logical or operator
    *
    * @param that a hardware $coll
    * @return the lgocial or of this $coll and `that`
    * @note this is equivalent to [[Bool!.|(that:chisel3\.Bool)* Bool.|)]]
    * @group Logical
    */
  def || (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_|| (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = (this | that).asBool

  /** Logical and operator
    *
    * @param that a hardware $coll
    * @return the lgocial and of this $coll and `that`
    * @note this is equivalent to [[Bool!.&(that:chisel3\.Bool)* Bool.&]]
    * @group Logical
    */
  def && (that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&& (that: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = (this & that).asBool

  /** Reinterprets this $coll as a clock */
  def asClock(): Clock = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asClock(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Clock = pushOp(DefPrim(sourceInfo, Clock(), AsClockOp, ref))
}

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
    extends Bits(width) with Num[FixedPoint] with NumBits[FixedPoint] {
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

    /** @group SourceInfoTransformMacro */
    final def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      FixedPoint.fromBigInt(0) - this
    /** @group SourceInfoTransformMacro */
    final def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      FixedPoint.fromBigInt(0) -% this

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

    private def expandingAddSub(that: FixedPoint, op: PrimOp, sourceInfo: SourceInfo) = {
      (this.width, that.width, this.binaryPoint, that.binaryPoint) match {
        case (KnownWidth(thisWidth), KnownWidth(thatWidth), KnownBinaryPoint(thisBP), KnownBinaryPoint(thatBP)) =>
          val thisIntWidth = thisWidth - thisBP
          val thatIntWidth = thatWidth - thatBP
          val newBinaryPoint = thisBP max thatBP
          val newWidth = (thisIntWidth max thatIntWidth) + newBinaryPoint + 1
          binop(sourceInfo, FixedPoint(newWidth.W, newBinaryPoint.BP), op, that)
        case _ =>
          val newBinaryPoint = this.binaryPoint max that.binaryPoint
          binop(sourceInfo, FixedPoint(UnknownWidth(), newBinaryPoint), op, that)
      }
    }
    /** @group SourceInfoTransformMacro */
    def do_+& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      expandingAddSub(that, AddOp, sourceInfo)

    /** @group SourceInfoTransformMacro */
    def do_+% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      (this +& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)
    /** @group SourceInfoTransformMacro */
    def do_-& (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      expandingAddSub(that, SubOp, sourceInfo)

    /** @group SourceInfoTransformMacro */
    def do_-% (that: FixedPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      (this -& that).tail(1).asFixedPoint(this.binaryPoint max that.binaryPoint)

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

    override def unary_~ (): FixedPoint = macro SourceInfoWhiteboxTransform.noArg

    /** @group SourceInfoTransformMacro */
    override def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint =
      throwException(s"Not is illegal on $this")

    // TODO(chick): Consider comparison with UInt and SInt

    override def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
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

    private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
      // TODO: redefine as just asFixedPoint on that, where FixedPoint.asFixedPoint just works.
      this := (that match {
        case fp: FixedPoint => fp.asSInt.asFixedPoint(this.binaryPoint)
        case _ => that.asUInt.asFixedPoint(this.binaryPoint)
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
        @deprecated("Use notation <double>.F(<binary_point>.BP) instead", "chisel3")
        def F(binaryPoint: Int): FixedPoint = FixedPoint.fromDouble(double, Width(), BinaryPoint(binaryPoint))

        def F(binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, Width(), binaryPoint)
        }

        def F(width: Width, binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, width, binaryPoint)
        }
      }
      // TODO extending AnyVal is faster but makes the class final and can't be done now because of the alias
      // FIXME after removal of the alias
      // implicit class fromIntToBinaryPoint(val int: Int) extends AnyVal {
      implicit class fromIntToBinaryPoint(val int: Int) {
        def BP: BinaryPoint = BinaryPoint(int) // scalastyle:ignore method.name
      }
    }
  }
}
