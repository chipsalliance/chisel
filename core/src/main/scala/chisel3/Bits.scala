// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{requireIsHardware, SourceInfo}
import chisel3.internal.{_resizeToWidth, throwException, BaseModule}
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.ir._
import chisel3.internal.firrtl.ir.PrimOp._
import _root_.firrtl.{ir => firrtlir}
import chisel3.internal.{castToInt, Builder, Warning, WarningID}
import chisel3.util.simpleClassName
import scala.annotation.nowarn

/** Data type for binary bit vectors
  *
  * The supertype for [[UInt]] and [[SInt]]. Note that the `Bits` factory method returns `UInts`.
  *
  * @define numType         hardware type
  * @define sumWidth        @note The width of the returned $numType is `width of this` + `width of that`.
  * @define sumWidthInt     @note The width of the returned $numType is `width of this` + `value of that`.
  * @define unchangedWidth  @note The width of the returned $numType is unchanged, i.e., the `width of this`.
  */
sealed abstract class Bits(private[chisel3] val width: Width) extends BitsIntf {

  // TODO: perhaps make this concrete?
  // Arguments for: self-checking code (can't do arithmetic on bits)
  // Arguments against: generates down to a FIRRTL UInt anyways

  // Only used for in a few cases, hopefully to be removed
  private[chisel3] def cloneTypeWidth(width: Width): this.type

  def cloneType: this.type = cloneTypeWidth(this.width)

  /** A non-ambiguous name of this `Bits` instance for use in generated Verilog names
    * Inserts the width directly after the typeName, e.g. UInt4, SInt1
    */
  override def typeName: String = s"${simpleClassName(this.getClass)}${this.width}"

  protected def _tailImpl(n: Int)(implicit sourceInfo: SourceInfo): UInt = {
    val w = this.width match {
      case KnownWidth(x) =>
        require(x >= n, s"Can't tail($n) for width $x < $n")
        Width(x - n)
      case UnknownWidth => Width()
    }
    binop(sourceInfo, UInt(width = w), TailOp, n)
  }

  protected def _headImpl(n: Int)(implicit sourceInfo: SourceInfo): UInt = {
    this.width match {
      case KnownWidth(x) => require(x >= n, s"Can't head($n) for width $x < $n")
      case UnknownWidth  => ()
    }
    binop(sourceInfo, UInt(Width(n)), HeadOp, n)
  }

  protected def _extractImpl(x: BigInt)(implicit sourceInfo: SourceInfo): Bool = {
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

  protected def _applyImpl(x: BigInt)(implicit sourceInfo: SourceInfo): Bool =
    _extractImpl(x)

  protected def _applyImpl(x: Int)(implicit sourceInfo: SourceInfo): Bool =
    _extractImpl(BigInt(x))

  protected def _takeImpl(n: Int)(implicit sourceInfo: SourceInfo): UInt = this._applyImpl(n - 1, 0)

  protected def _extractImpl(x: UInt)(implicit sourceInfo: SourceInfo): Bool = {
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

  protected def _applyImpl(x: UInt)(implicit sourceInfo: SourceInfo): Bool =
    _extractImpl(x)

  protected def _applyImpl(x: Int, y: Int)(implicit sourceInfo: SourceInfo): UInt = {
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

      // Illegal zero-width extractions are already caught, any at this point are legal.
      if (resultWidth != 0) {
        widthOption match {
          case Some(w) if w == 0 => Builder.error(s"Cannot extract from zero-width")
          case Some(w) if y >= w => Builder.error(s"High and low indices $x and $y are both out of range [0, ${w - 1}]")
          case Some(w) if x >= w => Builder.error(s"High index $x is out of range [0, ${w - 1}]")
          case _                 =>
        }
      }

      // FIRRTL does not yet support empty extraction so we must return the zero-width wire here:
      if (resultWidth == 0) {
        0.U(0.W)
      } else {
        pushOp(DefPrim(sourceInfo, UInt(Width(resultWidth)), BitsExtractOp, this.ref, ILit(x), ILit(y)))
      }
    }
  }

  protected def _applyImpl(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    _applyImpl(castToInt(x, "High index"), castToInt(y, "Low index"))

  private[chisel3] def unop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp): T = {
    implicit val info: SourceInfo = sourceInfo
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref))
  }
  private[chisel3] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: BigInt): T = {
    implicit val info: SourceInfo = sourceInfo
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, ILit(other)))
  }
  private[chisel3] def binop[T <: Data](sourceInfo: SourceInfo, dest: T, op: PrimOp, other: Bits): T = {
    implicit val info: SourceInfo = sourceInfo
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")
    pushOp(DefPrim(sourceInfo, dest, op, this.ref, other.ref))
  }
  private[chisel3] def compop(sourceInfo: SourceInfo, op: PrimOp, other: Bits): Bool = {
    implicit val info: SourceInfo = sourceInfo
    requireIsHardware(this, "bits operated on")
    requireIsHardware(other, "bits operated on")
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
  }
  private[chisel3] def redop(sourceInfo: SourceInfo, op: PrimOp): Bool = {
    implicit val info: SourceInfo = sourceInfo
    requireIsHardware(this, "bits operated on")
    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref))
  }

  // Pad literal to that width
  protected def _padLit(that: Int): this.type

  protected def _padImpl(that: Int)(implicit sourceInfo: SourceInfo): this.type = this.width match {
    case KnownWidth(w) if w >= that => this
    case _ if this.isLit            => this._padLit(that)
    case _                          => binop(sourceInfo, cloneTypeWidth(this.width.max(Width(that))), PadOp, that)
  }

  protected def _impl_unary_~(implicit sourceInfo: SourceInfo): Bits

  protected def _impl_<<(that: BigInt)(implicit sourceInfo: SourceInfo): Bits

  protected def _impl_<<(that: Int)(implicit sourceInfo: SourceInfo): Bits

  protected def _impl_<<(that: UInt)(implicit sourceInfo: SourceInfo): Bits

  protected def _impl_>>(that: BigInt)(implicit sourceInfo: SourceInfo): Bits

  protected def _impl_>>(that: Int)(implicit sourceInfo: SourceInfo): Bits

  protected def _impl_>>(that: UInt)(implicit sourceInfo: SourceInfo): Bits

  protected def _asBoolsImpl(implicit sourceInfo: SourceInfo): Seq[Bool] =
    Seq.tabulate(this.getWidth)(i => this(i))

  protected def _asSIntImpl(implicit sourceInfo: SourceInfo): SInt

  protected def _asBoolImpl(implicit sourceInfo: SourceInfo): Bool = {
    this.width match {
      case KnownWidth(1) => this(0)
      case _             => throwException(s"can't covert ${this.getClass.getSimpleName}${this.width} to Bool")
    }
  }

  protected def _impl_##(that: Bits)(implicit sourceInfo: SourceInfo): UInt = {
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

object Bits extends UIntFactory

/** A data type for unsigned integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[UInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class UInt private[chisel3] (width: Width) extends Bits(width) with UIntIntf with Num[UInt] {

  override def toString: String = {
    litOption match {
      case Some(value) => s"UInt$width($value)"
      case _           => stringAccessor(s"UInt$width")
    }
  }

  private[chisel3] override def cloneTypeWidth(w: Width): this.type =
    new UInt(w).asInstanceOf[this.type]

  override protected def _padLit(that: Int): this.type = {
    val value = this.litValue
    UInt.Lit(value, this.width.max(Width(that))).asInstanceOf[this.type]
  }

  protected def _impl_unary_-(implicit sourceInfo: SourceInfo): UInt = 0.U - this

  protected def _impl_unary_-%(implicit sourceInfo: SourceInfo): UInt = 0.U -% this

  protected def _impl_+(that: UInt)(implicit sourceInfo: SourceInfo): UInt = this +% that
  protected def _impl_-(that: UInt)(implicit sourceInfo: SourceInfo): UInt = this -% that
  protected def _impl_/(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width), DivideOp, that)
  protected def _impl_%(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.min(that.width)), RemOp, that)
  protected def _impl_*(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width + that.width), TimesOp, that)

  protected def _impl_*(that: SInt)(implicit sourceInfo: SourceInfo): SInt = that * this

  protected def _impl_+&(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt((this.width.max(that.width)) + 1), AddOp, that)

  protected def _impl_+%(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this +& that).tail(1)

  protected def _impl_-&(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this.subtractAsSInt(that)).asUInt

  protected def _impl_-%(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    (this.subtractAsSInt(that)).tail(1)

  protected def _absImpl(implicit sourceInfo: SourceInfo): UInt = this

  protected def _impl_&(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitAndOp, that)

  protected def _impl_|(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitOrOp, that)

  protected def _impl_^(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitXorOp, that)

  protected def _impl_unary_~(implicit sourceInfo: SourceInfo): UInt =
    unop(sourceInfo, UInt(width = width), BitNotOp)

  protected def _orRImpl(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, OrReduceOp)

  protected def _andRImpl(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, AndReduceOp)

  protected def _xorRImpl(implicit sourceInfo: SourceInfo): Bool = redop(sourceInfo, XorReduceOp)

  protected def _impl_<(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessOp, that)
  protected def _impl_>(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterOp, that)
  protected def _impl_<=(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessEqOp, that)
  protected def _impl_>=(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterEqOp, that)

  protected def _impl_=/=(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, NotEqualOp, that)

  protected def _impl_===(that: UInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, EqualOp, that)

  protected def _impl_unary_!(implicit sourceInfo: SourceInfo): Bool = this === 0.U(1.W)

  override protected def _impl_<<(that: Int)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override protected def _impl_<<(that: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    this << castToInt(that, "Shift amount")
  override protected def _impl_<<(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)

  // Implement legacy [buggy] UInt shr behavior for both Chisel and FIRRTL
  @nowarn("msg=method shiftRight in class Width is deprecated")
  private def legacyShiftRight(that: Int)(implicit sourceInfo: SourceInfo): UInt = {
    val resultWidth = this.width.shiftRight(that)
    val op = binop(sourceInfo, UInt(resultWidth), ShiftRightOp, validateShiftAmount(that))
    resultWidth match {
      // To emulate old FIRRTL behavior where minimum width is 1, we need to insert pad(_, 1) whenever
      // the width is or could be 0. Thus we check if it is known to be 0 or is unknown.
      case w @ (KnownWidth(0) | UnknownWidth) =>
        // Because we are inserting an extra op but we want stable emission (so the user can diff the output),
        // we need to seed a name to avoid name collisions.
        op.autoSeed("_shrLegacyWidthFixup")
        op.binop(sourceInfo, UInt(w), PadOp, 1)
      case _ => op
    }
  }

  override protected def _impl_>>(that: Int)(implicit sourceInfo: SourceInfo): UInt = {
    if (Builder.useLegacyWidth) legacyShiftRight(that)
    else binop(sourceInfo, UInt(this.width.unsignedShiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  }
  override protected def _impl_>>(that: BigInt)(implicit sourceInfo: SourceInfo): UInt =
    this >> castToInt(that, "Shift amount")
  override protected def _impl_>>(that: UInt)(implicit sourceInfo: SourceInfo): UInt =
    binop(sourceInfo, UInt(this.width), DynamicShiftRightOp, that)

  protected def _rotateLeftImpl(n: Int)(implicit sourceInfo: SourceInfo): UInt = width match {
    case _ if (n == 0)             => this
    case KnownWidth(w) if (w <= 1) => this
    case KnownWidth(w) if n >= w   => rotateLeft(n % w)
    case _ if (n < 0)              => rotateRight(-n)
    case _                         => tail(n) ## head(n)
  }

  protected def _rotateRightImpl(n: Int)(implicit sourceInfo: SourceInfo): UInt = width match {
    case _ if (n <= 0)             => rotateLeft(-n)
    case KnownWidth(w) if (w <= 1) => this
    case KnownWidth(w) if n >= w   => rotateRight(n % w)
    case _                         => this(n - 1, 0) ## (this >> n)
  }

  private def dynamicShift(
    n:           UInt,
    staticShift: (UInt, Int) => UInt
  )(
    implicit sourceInfo: SourceInfo
  ): UInt =
    n.asBools.zipWithIndex.foldLeft(this) { case (in, (en, sh)) =>
      Mux(en, staticShift(in, 1 << sh), in)
    }

  protected def _rotateRightImpl(n: UInt)(implicit sourceInfo: SourceInfo): UInt =
    dynamicShift(n, _ rotateRight _)

  protected def _rotateLeftImpl(n: UInt)(implicit sourceInfo: SourceInfo): UInt =
    dynamicShift(n, _ rotateLeft _)

  protected def _bitSetImpl(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo): UInt = {
    val bit = 1.U(1.W) << off
    Mux(dat, this | bit, ~(~this | bit))
  }

  protected def _zextImpl(implicit sourceInfo: SourceInfo): SInt = this.litOption match {
    case Some(value) => SInt.Lit(value, this.width + 1)
    case None        => pushOp(DefPrim(sourceInfo, SInt(width + 1), ConvertOp, ref))
  }

  override protected def _asSIntImpl(implicit sourceInfo: SourceInfo): SInt = this.litOption match {
    case Some(value) =>
      val w = this.width.get // Literals always have a known width, will be minimum legal width if not set
      val signedValue =
        // If width is 0, just return value (which will be 0).
        if (w > 0 && value.testBit(w - 1)) {
          // If the most significant bit is set, the SInt is negative and we need to adjust the value.
          value - (BigInt(1) << w)
        } else {
          value
        }
      // Using SInt.Lit instead of .S so we can use Width argument which may be Unknown
      SInt.Lit(signedValue, this.width.max(Width(1))) // SInt literal has width >= 1
    case None =>
      pushOp(DefPrim(sourceInfo, SInt(width), AsSIntOp, ref))
  }

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = this

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): this.type = {
    _resizeToWidth(that, this.widthOption, true)(identity).asInstanceOf[this.type]
  }

  private def subtractAsSInt(that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width.max(that.width)) + 1), SubOp, that)
}

object UInt extends UIntFactory

/** A data type for signed integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[SInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class SInt private[chisel3] (width: Width) extends Bits(width) with SIntIntf with Num[SInt] {
  override def toString: String = {
    litOption match {
      case Some(value) => s"SInt$width($value)"
      case _           => stringAccessor(s"SInt$width")
    }
  }

  private[chisel3] override def cloneTypeWidth(w: Width): this.type =
    new SInt(w).asInstanceOf[this.type]

  override protected def _padLit(that: Int): this.type = {
    val value = this.litValue
    SInt.Lit(value, this.width.max(Width(that))).asInstanceOf[this.type]
  }

  protected def _impl_unary_-(implicit sourceInfo: SourceInfo): SInt = 0.S - this

  protected def _impl_unary_-%(implicit sourceInfo: SourceInfo): SInt = 0.S -% this

  protected def _impl_+(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    this +% that

  protected def _impl_-(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    this -% that
  protected def _impl_*(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + that.width), TimesOp, that)
  protected def _impl_/(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + 1), DivideOp, that)
  protected def _impl_%(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width.min(that.width)), RemOp, that)

  protected def _impl_*(that: UInt)(implicit sourceInfo: SourceInfo): SInt = {
    val thatToSInt = that.zext
    val result = binop(sourceInfo, SInt(this.width + thatToSInt.width), TimesOp, thatToSInt)
    result.tail(1).asSInt
  }

  protected def _impl_+&(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width.max(that.width)) + 1), AddOp, that)

  protected def _impl_+%(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    (this +& that).tail(1).asSInt

  protected def _impl_-&(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt((this.width.max(that.width)) + 1), SubOp, that)

  protected def _impl_-%(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    (this -& that).tail(1).asSInt

  protected def _impl_&(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitAndOp, that).asSInt

  protected def _impl_|(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitOrOp, that).asSInt

  protected def _impl_^(that: SInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, UInt(this.width.max(that.width)), BitXorOp, that).asSInt

  protected def _impl_unary_~(implicit sourceInfo: SourceInfo): SInt =
    unop(sourceInfo, UInt(width = width), BitNotOp).asSInt

  protected def _impl_<(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessOp, that)
  protected def _impl_>(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterOp, that)
  protected def _impl_<=(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, LessEqOp, that)
  protected def _impl_>=(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, GreaterEqOp, that)

  protected def _impl_=/=(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, NotEqualOp, that)

  protected def _impl_===(that: SInt)(implicit sourceInfo: SourceInfo): Bool =
    compop(sourceInfo, EqualOp, that)

  protected def _absImpl(implicit sourceInfo: SourceInfo): SInt = {
    Mux(this < 0.S, -this, this)
  }

  override protected def _impl_<<(that: Int)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override protected def _impl_<<(that: BigInt)(implicit sourceInfo: SourceInfo): SInt =
    this << castToInt(that, "Shift amount")
  override protected def _impl_<<(that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)

  @nowarn("msg=method shiftRight in class Width is deprecated")
  override protected def _impl_>>(that: Int)(implicit sourceInfo: SourceInfo): SInt = {
    // We don't need to pad to emulate old behavior for SInt, just emulate old Chisel behavior with reported width.
    // FIRRTL will give a minimum of 1 bit for SInt.
    val newWidth = if (Builder.useLegacyWidth) this.width.shiftRight(that) else this.width.signedShiftRight(that)
    binop(sourceInfo, SInt(newWidth), ShiftRightOp, validateShiftAmount(that))
  }
  override protected def _impl_>>(that: BigInt)(implicit sourceInfo: SourceInfo): SInt =
    this >> castToInt(that, "Shift amount")
  override protected def _impl_>>(that: UInt)(implicit sourceInfo: SourceInfo): SInt =
    binop(sourceInfo, SInt(this.width), DynamicShiftRightOp, that)

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt =
    this.litOption match {
      case Some(value) =>
        // This is a reinterpretation of raw bits
        val posValue =
          if (value.signum == -1) {
            (BigInt(1) << this.width.get) + value
          } else {
            value
          }
        // Using UInt.Lit instead of .U so we can use Width argument which may be Unknown
        UInt.Lit(posValue, this.width)
      case None =>
        pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
    }

  override def _asSIntImpl(implicit sourceInfo: SourceInfo): SInt = this

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): this.type =
    _resizeToWidth(that.asSInt, this.widthOption, false)(_.asSInt).asInstanceOf[this.type]
}

object SInt extends SIntFactory

sealed trait Reset extends Element with ResetIntf {

  protected def _asAsyncResetImpl(implicit sourceInfo: SourceInfo): AsyncReset

  protected def _asDisableImpl(implicit sourceInfo: SourceInfo): Disable = new Disable(this.asBool)
}

object Reset {
  def apply(): Reset = new ResetType
}

/** "Abstract" Reset Type inferred in FIRRTL to either [[AsyncReset]] or [[Bool]]
  *
  * @note This shares a common interface with [[AsyncReset]] and [[Bool]] but is not their actual
  * super type due to Bool inheriting from abstract class UInt
  */
final class ResetType(private[chisel3] val width: Width = Width(1)) extends Reset with ResetTypeIntf {

  override def toString: String = stringAccessor("Reset")

  def cloneType: this.type = Reset().asInstanceOf[this.type]

  override def litOption: Option[BigInt] = None

  /** Not really supported */
  def toPrintable: Printable = PString("Reset")

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = pushOp(
    DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref)
  )

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data = {
    val _wire = Wire(this.cloneTypeFull)
    _wire := that
    _wire
  }

  protected def _asAsyncResetImpl(implicit sourceInfo: SourceInfo): AsyncReset =
    pushOp(DefPrim(sourceInfo, AsyncReset(), AsAsyncResetOp, ref))

  protected def _asBoolImpl(implicit sourceInfo: SourceInfo): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), AsUIntOp, ref))
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
sealed class AsyncReset(private[chisel3] val width: Width = Width(1)) extends Element with AsyncResetIntf with Reset {

  override def toString: String = stringAccessor("AsyncReset")

  def cloneType: this.type = AsyncReset().asInstanceOf[this.type]

  override def litOption: Option[BigInt] = None

  /** Not really supported */
  def toPrintable: Printable = PString("AsyncReset")

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = pushOp(
    DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref)
  )

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data = that.asBool.asAsyncReset

  protected def _asAsyncResetImpl(implicit sourceInfo: SourceInfo): AsyncReset = this

  protected def _asBoolImpl(implicit sourceInfo: SourceInfo): Bool =
    pushOp(DefPrim(sourceInfo, Bool(), AsUIntOp, ref))
}

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  *
  * @define coll [[Bool]]
  * @define numType $coll
  */
sealed class Bool() extends UInt(1.W) with BoolIntf with Reset {

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

  protected def _impl_&(that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitAndOp, that)

  protected def _impl_|(that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitOrOp, that)

  protected def _impl_^(that: Bool)(implicit sourceInfo: SourceInfo): Bool =
    binop(sourceInfo, Bool(), BitXorOp, that)

  override protected def _impl_unary_~(implicit sourceInfo: SourceInfo): Bool =
    unop(sourceInfo, Bool(), BitNotOp)

  protected def _impl_||(that: Bool)(implicit sourceInfo: SourceInfo): Bool = this | that

  protected def _impl_&&(that: Bool)(implicit sourceInfo: SourceInfo): Bool = this & that

  override protected def _asBoolImpl(implicit sourceInfo: SourceInfo): Bool = this

  protected def _asClockImpl(implicit sourceInfo: SourceInfo): Clock = pushOp(
    DefPrim(sourceInfo, Clock(), AsClockOp, ref)
  )

  protected def _asAsyncResetImpl(implicit sourceInfo: SourceInfo): AsyncReset =
    pushOp(DefPrim(sourceInfo, AsyncReset(), AsAsyncResetOp, ref))

  protected def _asResetImpl(implicit sourceInfo: SourceInfo): Reset =
    pushOp(DefPrim(sourceInfo, Reset(), AsResetOp, ref))

  protected def _impl_implies(that: Bool)(implicit sourceInfo: SourceInfo): Bool = (!this) | that

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): this.type = {
    _resizeToWidth(that, this.widthOption, true)(identity).asBool.asInstanceOf[this.type]
  }
}

object Bool extends BoolFactory
