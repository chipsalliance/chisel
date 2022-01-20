// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl

import firrtl.{ir => fir}
import chisel3._
import chisel3.internal._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental._
import _root_.firrtl.{ir => firrtlir}
import _root_.firrtl.{PrimOps, RenameMap}
import _root_.firrtl.annotations.Annotation

import scala.collection.immutable.NumericRange
import scala.math.BigDecimal.RoundingMode
import scala.annotation.nowarn


case class PrimOp(name: String) {
  override def toString: String = name
}

object PrimOp {
  val AddOp = PrimOp("add")
  val SubOp = PrimOp("sub")
  val TailOp = PrimOp("tail")
  val HeadOp = PrimOp("head")
  val TimesOp = PrimOp("mul")
  val DivideOp = PrimOp("div")
  val RemOp = PrimOp("rem")
  val ShiftLeftOp = PrimOp("shl")
  val ShiftRightOp = PrimOp("shr")
  val DynamicShiftLeftOp = PrimOp("dshl")
  val DynamicShiftRightOp = PrimOp("dshr")
  val BitAndOp = PrimOp("and")
  val BitOrOp = PrimOp("or")
  val BitXorOp = PrimOp("xor")
  val BitNotOp = PrimOp("not")
  val ConcatOp = PrimOp("cat")
  val BitsExtractOp = PrimOp("bits")
  val LessOp = PrimOp("lt")
  val LessEqOp = PrimOp("leq")
  val GreaterOp = PrimOp("gt")
  val GreaterEqOp = PrimOp("geq")
  val EqualOp = PrimOp("eq")
  val PadOp = PrimOp("pad")
  val NotEqualOp = PrimOp("neq")
  val NegOp = PrimOp("neg")
  val MultiplexOp = PrimOp("mux")
  val AndReduceOp = PrimOp("andr")
  val OrReduceOp = PrimOp("orr")
  val XorReduceOp = PrimOp("xorr")
  val ConvertOp = PrimOp("cvt")
  val AsUIntOp = PrimOp("asUInt")
  val AsSIntOp = PrimOp("asSInt")
  val AsFixedPointOp = PrimOp("asFixedPoint")
  val AsIntervalOp = PrimOp("asInterval")
  val WrapOp = PrimOp("wrap")
  val SqueezeOp = PrimOp("squz")
  val ClipOp = PrimOp("clip")
  val SetBinaryPoint = PrimOp("setp")
  val IncreasePrecision = PrimOp("incp")
  val DecreasePrecision = PrimOp("decp")
  val AsClockOp = PrimOp("asClock")
  val AsAsyncResetOp = PrimOp("asAsyncReset")
}

sealed abstract class Arg {
  def localName: String = name
  def contextualName(ctx: Component): String = name
  def fullName(ctx: Component): String = contextualName(ctx)
  def name: String
}

case class Node(id: HasId) extends Arg {
  override def contextualName(ctx: Component): String = id.getOptionRef match {
    case Some(arg) => arg.contextualName(ctx)
    case None => id.instanceName
  }
  override def localName: String = id.getOptionRef match {
    case Some(arg) => arg.localName
    case None => id.instanceName
  }
  def name: String = id.getOptionRef match {
    case Some(arg) => arg.name
    case None => id.instanceName
  }
}

object Arg {
  def earlyLocalName(id: HasId): String = id.getOptionRef match {
    case Some(Index(Node(imm), Node(value))) => s"${earlyLocalName(imm)}[${earlyLocalName(imm)}]"
    case Some(Index(Node(imm), arg)) => s"${earlyLocalName(imm)}[${arg.localName}]"
    case Some(Slot(Node(imm), name)) => s"${earlyLocalName(imm)}.$name"
    case Some(arg) => arg.name
    case None => id match {
      case data: Data => data._computeName(None, Some("?")).get
      case _ => "?"
    }
  }
}

abstract class LitArg(val num: BigInt, widthArg: Width) extends Arg {
  private[chisel3] def forcedWidth = widthArg.known
  private[chisel3] def width: Width = if (forcedWidth) widthArg else Width(minWidth)
  override def contextualName(ctx: Component): String = name
  // Ensure the node representing this LitArg has a ref to it and a literal binding.
  def bindLitArg[T <: Element](elem: T): T = {
    elem.bind(ElementLitBinding(this))
    elem.setRef(this)
    elem
  }

  /** Provides a mechanism that LitArgs can have their width adjusted
    * to match other members of a VecLiteral
    *
    * @param newWidth the new width for this
    * @return
    */
  def cloneWithWidth(newWidth: Width): this.type

  protected def minWidth: Int
  if (forcedWidth) {
    require(widthArg.get >= minWidth,
      s"The literal value ${num} was elaborated with a specified width of ${widthArg.get} bits, but at least ${minWidth} bits are required.")
  }
}

case class ILit(n: BigInt) extends Arg {
  def name: String = n.toString
}

case class ULit(n: BigInt, w: Width) extends LitArg(n, w) {
  def name: String = "UInt" + width + "(\"h0" + num.toString(16) + "\")"
  def minWidth: Int = 1 max n.bitLength

  def cloneWithWidth(newWidth: Width): this.type = {
    ULit(n, newWidth).asInstanceOf[this.type]
  }

  require(n >= 0, s"UInt literal ${n} is negative")
}

case class SLit(n: BigInt, w: Width) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asSInt(${ULit(unsigned, width).name})"
  }
  def minWidth: Int = 1 + n.bitLength

  def cloneWithWidth(newWidth: Width): this.type = {
    SLit(n, newWidth).asInstanceOf[this.type]
  }
}

case class FPLit(n: BigInt, w: Width, binaryPoint: BinaryPoint) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asFixedPoint(${ULit(unsigned, width).name}, ${binaryPoint.asInstanceOf[KnownBinaryPoint].value})"
  }
  def minWidth: Int = 1 + n.bitLength

  def cloneWithWidth(newWidth: Width): this.type = {
    FPLit(n, newWidth, binaryPoint).asInstanceOf[this.type]
  }
}

case class IntervalLit(n: BigInt, w: Width, binaryPoint: BinaryPoint) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asInterval(${ULit(unsigned, width).name}, ${n}, ${n}, ${binaryPoint.asInstanceOf[KnownBinaryPoint].value})"
  }
  val range: IntervalRange = {
    new IntervalRange(IntervalRange.getBound(isClosed = true, BigDecimal(n)),
      IntervalRange.getBound(isClosed = true, BigDecimal(n)), IntervalRange.getRangeWidth(binaryPoint))
  }
  def minWidth: Int = 1 + n.bitLength

  def cloneWithWidth(newWidth: Width): this.type = {
    IntervalLit(n, newWidth, binaryPoint).asInstanceOf[this.type]
  }
}

case class Ref(name: String) extends Arg
/** Arg for ports of Modules
  * @param mod the module this port belongs to
  * @param name the name of the port
  */
case class ModuleIO(mod: BaseModule, name: String) extends Arg {
  override def contextualName(ctx: Component): String =
    if (mod eq ctx.id) name else s"${mod.getRef.name}.$name"
}
/** Ports of cloned modules (CloneModuleAsRecord)
  * @param mod The original module for which these ports are a clone
  * @param name the name of the module instance
  */
case class ModuleCloneIO(mod: BaseModule, name: String) extends Arg {
  override def localName = ""
  override def contextualName(ctx: Component): String =
    // NOTE: mod eq ctx.id only occurs in Target and Named-related APIs
    if (mod eq ctx.id) localName else name
}
case class Slot(imm: Node, name: String) extends Arg {
  override def contextualName(ctx: Component): String = {
    val immName = imm.contextualName(ctx)
    if (immName.isEmpty) name else s"$immName.$name"
  }
  override def localName: String = {
    val immName = imm.localName
    if (immName.isEmpty) name else s"$immName.$name"
  }
}

case class Index(imm: Arg, value: Arg) extends Arg {
  def name: String = s"[$value]"
  override def contextualName(ctx: Component): String = s"${imm.contextualName(ctx)}[${value.contextualName(ctx)}]"
  override def localName: String = s"${imm.localName}[${value.localName}]"
}

object Width {
  def apply(x: Int): Width = KnownWidth(x)
  def apply(): Width = UnknownWidth()
}

sealed abstract class Width {
  type W = Int
  def min(that: Width): Width = this.op(that, _ min _)
  def max(that: Width): Width = this.op(that, _ max _)
  def + (that: Width): Width = this.op(that, _ + _)
  def + (that: Int): Width = this.op(this, (a, b) => a + that)
  def shiftRight(that: Int): Width = this.op(this, (a, b) => 0 max (a - that))
  def dynamicShiftLeft(that: Width): Width =
    this.op(that, (a, b) => a + (1 << b) - 1)

  def known: Boolean
  def get: W
  protected def op(that: Width, f: (W, W) => W): Width
}

sealed case class UnknownWidth() extends Width {
  def known: Boolean = false
  def get: Int = None.get
  def op(that: Width, f: (W, W) => W): Width = this
  override def toString: String = ""
}

sealed case class KnownWidth(value: Int) extends Width {
  require(value >= 0)
  def known: Boolean = true
  def get: Int = value
  def op(that: Width, f: (W, W) => W): Width = that match {
    case KnownWidth(x) => KnownWidth(f(value, x))
    case _ => that
  }
  override def toString: String = s"<${value.toString}>"
}

object BinaryPoint {
  def apply(x: Int): BinaryPoint = KnownBinaryPoint(x)
  def apply(): BinaryPoint = UnknownBinaryPoint
}

sealed abstract class BinaryPoint {
  type W = Int
  def max(that: BinaryPoint): BinaryPoint = this.op(that, _ max _)
  def + (that: BinaryPoint): BinaryPoint = this.op(that, _ + _)
  def + (that: Int): BinaryPoint = this.op(this, (a, b) => a + that)
  def shiftRight(that: Int): BinaryPoint = this.op(this, (a, b) => 0 max (a - that))
  def dynamicShiftLeft(that: BinaryPoint): BinaryPoint =
    this.op(that, (a, b) => a + (1 << b) - 1)

  def known: Boolean
  def get: W
  protected def op(that: BinaryPoint, f: (W, W) => W): BinaryPoint
}

case object UnknownBinaryPoint extends BinaryPoint {
  def known: Boolean = false
  def get: Int = None.get
  def op(that: BinaryPoint, f: (W, W) => W): BinaryPoint = this
  override def toString: String = ""
}

sealed case class KnownBinaryPoint(value: Int) extends BinaryPoint {
  def known: Boolean = true
  def get: Int = value
  def op(that: BinaryPoint, f: (W, W) => W): BinaryPoint = that match {
    case KnownBinaryPoint(x) => KnownBinaryPoint(f(value, x))
    case _ => that
  }
  override def toString: String = s"<<${value.toString}>>"
}


sealed abstract class MemPortDirection(name: String) {
  override def toString: String = name
}
object MemPortDirection {
  object READ extends MemPortDirection("read")
  object WRITE extends MemPortDirection("write")
  object RDWR extends MemPortDirection("rdwr")
  object INFER extends MemPortDirection("infer")
}

sealed trait RangeType {
  def getWidth: Width

  def * (that: IntervalRange): IntervalRange
  def +& (that: IntervalRange): IntervalRange
  def -& (that: IntervalRange): IntervalRange
  def << (that: Int): IntervalRange
  def >> (that: Int): IntervalRange
  def << (that: KnownWidth): IntervalRange
  def >> (that: KnownWidth): IntervalRange
  def merge(that: IntervalRange): IntervalRange
}

object IntervalRange {
  /** Creates an IntervalRange, this is used primarily by the range interpolator macro
    * @param lower               lower bound
    * @param upper               upper bound
    * @param firrtlBinaryPoint   binary point firrtl style
    * @return
    */
  def apply(lower: firrtlir.Bound, upper: firrtlir.Bound, firrtlBinaryPoint: firrtlir.Width): IntervalRange = {
    new IntervalRange(lower, upper, firrtlBinaryPoint)
  }

  def apply(lower: firrtlir.Bound, upper: firrtlir.Bound, binaryPoint: BinaryPoint): IntervalRange = {
    new IntervalRange(lower, upper, IntervalRange.getBinaryPoint(binaryPoint))
  }

  def apply(lower: firrtlir.Bound, upper: firrtlir.Bound, binaryPoint: Int): IntervalRange = {
    IntervalRange(lower, upper, BinaryPoint(binaryPoint))
  }

  /** Returns an IntervalRange appropriate for a signed value of the given width
    * @param binaryPoint  number of bits of mantissa
    * @return
    */
  def apply(binaryPoint: BinaryPoint): IntervalRange = {
    IntervalRange(firrtlir.UnknownBound, firrtlir.UnknownBound, binaryPoint)
  }

  /** Returns an IntervalRange appropriate for a signed value of the given width
    * @param width        number of bits to have in the interval
    * @param binaryPoint  number of bits of mantissa
    * @return
    */
  def apply(width: Width, binaryPoint: BinaryPoint = 0.BP): IntervalRange = {
    val range = width match {
      case KnownWidth(w) =>
        val nearestPowerOf2 = BigInt("1" + ("0" * (w - 1)), 2)
        IntervalRange(
          firrtlir.Closed(BigDecimal(-nearestPowerOf2)), firrtlir.Closed(BigDecimal(nearestPowerOf2 - 1)), binaryPoint
        )
      case _ =>
        IntervalRange(firrtlir.UnknownBound, firrtlir.UnknownBound, binaryPoint)
    }
    range
  }

  def unapply(arg: IntervalRange): Option[(firrtlir.Bound, firrtlir.Bound, BinaryPoint)] = {
    return Some((arg.lower, arg.upper, arg.binaryPoint))
  }

  def getBound(isClosed: Boolean, value: String): firrtlir.Bound = {
    if(value == "?") {
      firrtlir.UnknownBound
    }
    else if(isClosed) {
      firrtlir.Closed(BigDecimal(value))
    }
    else {
      firrtlir.Open(BigDecimal(value))
    }
  }

  def getBound(isClosed: Boolean, value: BigDecimal): firrtlir.Bound = {
    if(isClosed) {
      firrtlir.Closed(value)
    }
    else {
      firrtlir.Open(value)
    }
  }

  def getBound(isClosed: Boolean, value: Int): firrtlir.Bound = {
    getBound(isClosed, (BigDecimal(value)))
  }

  def getBinaryPoint(s: String): firrtlir.Width = {
    firrtlir.UnknownWidth
  }

  def getBinaryPoint(n: Int): firrtlir.Width = {
    if(n < 0) {
      firrtlir.UnknownWidth
    }
    else {
      firrtlir.IntWidth(n)
    }
  }
  def getBinaryPoint(n: BinaryPoint): firrtlir.Width = {
    n match {
      case UnknownBinaryPoint => firrtlir.UnknownWidth
      case KnownBinaryPoint(w) => firrtlir.IntWidth(w)
    }
  }

  def getRangeWidth(w: Width): firrtlir.Width = {
    if(w.known) {
      firrtlir.IntWidth(w.get)
    }
    else {
      firrtlir.UnknownWidth
    }
  }
  def getRangeWidth(binaryPoint: BinaryPoint): firrtlir.Width = {
    if(binaryPoint.known) {
      firrtlir.IntWidth(binaryPoint.get)
    }
    else {
      firrtlir.UnknownWidth
    }
  }

  def Unknown: IntervalRange = range"[?,?].?"
}


sealed class IntervalRange(
                            val lowerBound: firrtlir.Bound,
                            val upperBound: firrtlir.Bound,
                            private[chisel3] val firrtlBinaryPoint: firrtlir.Width)
  extends firrtlir.IntervalType(lowerBound, upperBound, firrtlBinaryPoint)
    with RangeType {

  (lowerBound, upperBound) match {
    case (firrtlir.Open(begin), firrtlir.Open(end)) =>
      if(begin >= end) throw new ChiselException(s"Invalid range with ${serialize}")
      binaryPoint match {
        case KnownBinaryPoint(bp) =>
          if(begin >= end - (BigDecimal(1) / BigDecimal(BigInt(1) << bp))) {
            throw new ChiselException(s"Invalid range with ${serialize}")
          }
        case _ =>
      }
    case (firrtlir.Open(begin), firrtlir.Closed(end)) =>
      if(begin >= end) throw new ChiselException(s"Invalid range with ${serialize}")
    case (firrtlir.Closed(begin), firrtlir.Open(end)) =>
      if(begin >= end) throw new ChiselException(s"Invalid range with ${serialize}")
    case (firrtlir.Closed(begin), firrtlir.Closed(end)) =>
      if(begin > end) throw new ChiselException(s"Invalid range with ${serialize}")
    case _ =>
  }

  override def toString: String = {
    val binaryPoint = firrtlBinaryPoint match {
      case firrtlir.IntWidth(n) => s"$n"
      case _ => "?"
    }
    val lowerBoundString = lowerBound match {
      case firrtlir.Closed(l)      => s"[$l"
      case firrtlir.Open(l)        => s"($l"
      case firrtlir.UnknownBound   => s"[?"
    }
    val upperBoundString = upperBound match {
      case firrtlir.Closed(l)      => s"$l]"
      case firrtlir.Open(l)        => s"$l)"
      case firrtlir.UnknownBound   => s"?]"
    }
    s"""range"$lowerBoundString,$upperBoundString.$binaryPoint""""
  }

  val increment: Option[BigDecimal] = firrtlBinaryPoint match {
    case firrtlir.IntWidth(bp) =>
      Some(BigDecimal(math.pow(2, -bp.doubleValue)))
    case _ => None
  }

  /** If possible returns the lowest possible value for this Interval
    * @return
    */
  val getLowestPossibleValue: Option[BigDecimal] = {
    increment match {
      case Some(inc) =>
        lower match {
          case firrtlir.Closed(n) => Some(n)
          case firrtlir.Open(n) => Some(n + inc)
          case _ => None
        }
      case _ =>
        None
    }
  }

  /** If possible returns the highest possible value for this Interval
    * @return
    */
  val getHighestPossibleValue: Option[BigDecimal] = {
    increment match {
      case Some(inc) =>
        upper match {
          case firrtlir.Closed(n) => Some(n)
          case firrtlir.Open(n) => Some(n - inc)
          case _ => None
        }
      case _ =>
        None
    }
  }

  /** Return a Seq of the possible values for this range
    * Mostly to be used for testing
    * @return
    */
  def getPossibleValues: NumericRange[BigDecimal] = {
    (getLowestPossibleValue, getHighestPossibleValue, increment) match {
      case (Some(low), Some(high), Some(inc)) => (low to high by inc)
      case (_, _, None) =>
        throw new ChiselException(s"BinaryPoint unknown. Cannot get possible values from IntervalRange $toString")
      case _ =>
        throw new ChiselException(s"Unknown Bound. Cannot get possible values from IntervalRange $toString")

    }
  }

  override def getWidth: Width = {
    width match {
      case firrtlir.IntWidth(n) => KnownWidth(n.toInt)
      case firrtlir.UnknownWidth => UnknownWidth()
    }
  }

  private def doFirrtlOp(op: firrtlir.PrimOp, that: IntervalRange): IntervalRange = {
    PrimOps.set_primop_type(
      firrtlir.DoPrim(op,
        Seq(firrtlir.Reference("a", this), firrtlir.Reference("b", that)), Nil,firrtlir.UnknownType)
    ).tpe match {
      case i: firrtlir.IntervalType => IntervalRange(i.lower, i.upper, i.point)
      case other => sys.error("BAD!")
    }
  }

  private def doFirrtlDynamicShift(that: UInt, isLeft: Boolean): IntervalRange = {
    val uinttpe = that.widthOption match {
      case None => firrtlir.UIntType(firrtlir.UnknownWidth)
      case Some(w) => firrtlir.UIntType(firrtlir.IntWidth(w))
    }
    val op = if(isLeft) PrimOps.Dshl else PrimOps.Dshr
    PrimOps.set_primop_type(
      firrtlir.DoPrim(op,
        Seq(firrtlir.Reference("a", this), firrtlir.Reference("b", uinttpe)), Nil,firrtlir.UnknownType)
    ).tpe match {
      case i: firrtlir.IntervalType => IntervalRange(i.lower, i.upper, i.point)
      case other => sys.error("BAD!")
    }
  }

  private def doFirrtlOp(op: firrtlir.PrimOp, that: Int): IntervalRange = {
    PrimOps.set_primop_type(
      firrtlir.DoPrim(op,
        Seq(firrtlir.Reference("a", this)), Seq(BigInt(that)), firrtlir.UnknownType)
    ).tpe match {
      case i: firrtlir.IntervalType => IntervalRange(i.lower, i.upper, i.point)
      case other => sys.error("BAD!")
    }
  }

  /** Multiply this by that, here we return a fully unknown range,
    * firrtl's range inference can figure this out
    * @param that
    * @return
    */
  override def *(that: IntervalRange): IntervalRange = {
    doFirrtlOp(PrimOps.Mul, that)
  }

  /** Add that to this, here we return a fully unknown range,
    * firrtl's range inference can figure this out
    * @param that
    * @return
    */
  override def +&(that: IntervalRange): IntervalRange = {
    doFirrtlOp(PrimOps.Add, that)
  }

  /** Subtract that from this, here we return a fully unknown range,
    * firrtl's range inference can figure this out
    * @param that
    * @return
    */
  override def -&(that: IntervalRange): IntervalRange = {
    doFirrtlOp(PrimOps.Sub, that)
  }

  private def adjustBoundValue(value: BigDecimal, binaryPointValue: Int): BigDecimal = {
    if(binaryPointValue >= 0) {
      val maskFactor = BigDecimal(1 << binaryPointValue)
      val a = (value * maskFactor)
      val b = a.setScale(0, RoundingMode.DOWN)
      val c = b / maskFactor
      c
    } else {
      value
    }
  }

  private def adjustBound(bound: firrtlir.Bound, binaryPoint: BinaryPoint): firrtlir.Bound = {
    binaryPoint match {
      case KnownBinaryPoint(binaryPointValue) =>
        bound match {
          case firrtlir.Open(value) => firrtlir.Open(adjustBoundValue(value, binaryPointValue))
          case firrtlir.Closed(value) => firrtlir.Closed(adjustBoundValue(value, binaryPointValue))
          case _ => bound
        }
      case _ => firrtlir.UnknownBound
    }
  }

  /** Creates a new range with the increased precision
    *
    * @param newBinaryPoint
    * @return
    */
  def incPrecision(newBinaryPoint: BinaryPoint): IntervalRange = {
    newBinaryPoint match {
      case KnownBinaryPoint(that) =>
        doFirrtlOp(PrimOps.IncP, that)
      case _ =>
        throwException(s"$this.incPrecision(newBinaryPoint = $newBinaryPoint) error, newBinaryPoint must be know")
    }
  }

  /** Creates a new range with the decreased precision
    *
    * @param newBinaryPoint
    * @return
    */
  def decPrecision(newBinaryPoint: BinaryPoint): IntervalRange = {
    newBinaryPoint match {
      case KnownBinaryPoint(that) =>
        doFirrtlOp(PrimOps.DecP, that)
      case _ =>
        throwException(s"$this.decPrecision(newBinaryPoint = $newBinaryPoint) error, newBinaryPoint must be know")
    }
  }

  /** Creates a new range with the given binary point, adjusting precision
    * on bounds as necessary
    *
    * @param newBinaryPoint
    * @return
    */
  def setPrecision(newBinaryPoint: BinaryPoint): IntervalRange = {
    newBinaryPoint match {
      case KnownBinaryPoint(that) =>
        doFirrtlOp(PrimOps.SetP, that)
      case _ =>
        throwException(s"$this.setPrecision(newBinaryPoint = $newBinaryPoint) error, newBinaryPoint must be know")
    }
  }

  /** Shift this range left, i.e. shifts the min and max by the specified amount
    * @param that
    * @return
    */
  override def <<(that: Int): IntervalRange = {
    doFirrtlOp(PrimOps.Shl, that)
  }

  /** Shift this range left, i.e. shifts the min and max by the known width
    * @param that
    * @return
    */
  override def <<(that: KnownWidth): IntervalRange = {
    <<(that.value)
  }

  /** Shift this range left, i.e. shifts the min and max by value
    * @param that
    * @return
    */
  def <<(that: UInt): IntervalRange = {
    doFirrtlDynamicShift(that, isLeft = true)
  }

  /** Shift this range right, i.e. shifts the min and max by the specified amount
    * @param that
    * @return
    */
  override def >>(that: Int): IntervalRange = {
    doFirrtlOp(PrimOps.Shr, that)
  }

  /** Shift this range right, i.e. shifts the min and max by the known width
    * @param that
    * @return
    */
  override def >>(that: KnownWidth): IntervalRange = {
    >>(that.value)
  }

  /** Shift this range right, i.e. shifts the min and max by value
    * @param that
    * @return
    */
  def >>(that: UInt): IntervalRange = {
    doFirrtlDynamicShift(that, isLeft = false)
  }

  /**
    * Squeeze returns the intersection of the ranges this interval and that Interval
    * @param that
    * @return
    */
  def squeeze(that: IntervalRange): IntervalRange = {
    doFirrtlOp(PrimOps.Squeeze, that)
  }

  /**
    * Wrap the value of this [[Interval]] into the range of a different Interval with a presumably smaller range.
    * @param that
    * @return
    */
  def wrap(that: IntervalRange): IntervalRange = {
    doFirrtlOp(PrimOps.Wrap, that)
  }

  /**
    * Clip the value of this [[Interval]] into the range of a different Interval with a presumably smaller range.
    * @param that
    * @return
    */
  def clip(that: IntervalRange): IntervalRange = {
    doFirrtlOp(PrimOps.Clip, that)
  }

  /** merges the ranges of this and that, basically takes lowest low, highest high and biggest bp
    * set unknown if any of this or that's value of above is unknown
    * Like an union but will slurp up points in between the two ranges that were part of neither
    * @param that
    * @return
    */
  override def merge(that: IntervalRange): IntervalRange = {
    val lowest = (this.getLowestPossibleValue, that.getLowestPossibleValue) match {
      case (Some(l1), Some(l2)) =>
        if(l1 < l2) { this.lower } else { that.lower }
      case _ =>
        firrtlir.UnknownBound
    }
    val highest = (this.getHighestPossibleValue, that.getHighestPossibleValue) match {
      case (Some(l1), Some(l2)) =>
        if(l1 >= l2) { this.lower } else { that.lower }
      case _ =>
        firrtlir.UnknownBound
    }
    val newBinaryPoint = (this.firrtlBinaryPoint, that.firrtlBinaryPoint) match {
      case (firrtlir.IntWidth(b1), firrtlir.IntWidth(b2)) =>
        if(b1 > b2) { firrtlir.IntWidth(b1)} else { firrtlir.IntWidth(b2) }
      case _ =>
        firrtlir.UnknownWidth
    }
    IntervalRange(lowest, highest, newBinaryPoint)
  }

  def binaryPoint: BinaryPoint = {
    firrtlBinaryPoint match {
      case firrtlir.IntWidth(n) =>
        assert(n < Int.MaxValue, s"binary point value $n is out of range")
        KnownBinaryPoint(n.toInt)
      case _ => UnknownBinaryPoint
    }
  }
}

abstract class Command {
  def sourceInfo: SourceInfo
}
abstract class Definition extends Command {
  def id: HasId
  def name: String = id.getRef.name
}
case class DefPrim[T <: Data](sourceInfo: SourceInfo, id: T, op: PrimOp, args: Arg*) extends Definition
case class DefInvalid(sourceInfo: SourceInfo, arg: Arg) extends Command
case class DefWire(sourceInfo: SourceInfo, id: Data) extends Definition
case class DefReg(sourceInfo: SourceInfo, id: Data, clock: Arg) extends Definition
case class DefRegInit(sourceInfo: SourceInfo, id: Data, clock: Arg, reset: Arg, init: Arg) extends Definition
case class DefMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt) extends Definition
case class DefSeqMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: BigInt, readUnderWrite: fir.ReadUnderWrite.Value) extends Definition
case class DefMemPort[T <: Data](sourceInfo: SourceInfo, id: T, source: Node, dir: MemPortDirection, index: Arg, clock: Arg) extends Definition
@nowarn("msg=class Port") // delete when Port becomes private
case class DefInstance(sourceInfo: SourceInfo, id: BaseModule, ports: Seq[Port]) extends Definition
case class WhenBegin(sourceInfo: SourceInfo, pred: Arg) extends Command
case class WhenEnd(sourceInfo: SourceInfo, firrtlDepth: Int, hasAlt: Boolean = false) extends Command
case class AltBegin(sourceInfo: SourceInfo) extends Command
case class OtherwiseEnd(sourceInfo: SourceInfo, firrtlDepth: Int) extends Command
case class Connect(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
case class BulkConnect(sourceInfo: SourceInfo, loc1: Node, loc2: Node) extends Command
case class Attach(sourceInfo: SourceInfo, locs: Seq[Node]) extends Command
case class ConnectInit(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
case class Stop(id: stop.Stop, sourceInfo: SourceInfo, clock: Arg, ret: Int) extends Definition
// Note this is just deprecated which will cause deprecation warnings, use @nowarn
@deprecated("This API should never have been public, for Module port reflection, use DataMirror.modulePorts", "Chisel 3.5")
case class Port(id: Data, dir: SpecifiedDirection)
case class Printf(id: printf.Printf, sourceInfo: SourceInfo, clock: Arg, pable: Printable) extends Definition
object Formal extends Enumeration {
  val Assert = Value("assert")
  val Assume = Value("assume")
  val Cover = Value("cover")
}
case class Verification[T <: VerificationStatement](id: T, op: Formal.Value, sourceInfo: SourceInfo, clock: Arg,
                        predicate: Arg, message: String) extends Definition
@nowarn("msg=class Port") // delete when Port becomes private
abstract class Component extends Arg {
  def id: BaseModule
  def name: String
  def ports: Seq[Port]
}
@nowarn("msg=class Port") // delete when Port becomes private
case class DefModule(id: RawModule, name: String, ports: Seq[Port], commands: Seq[Command]) extends Component
@nowarn("msg=class Port") // delete when Port becomes private
case class DefBlackBox(id: BaseBlackBox, name: String, ports: Seq[Port], topDir: SpecifiedDirection, params: Map[String, Param]) extends Component

case class Circuit(name: String, components: Seq[Component], annotations: Seq[ChiselAnnotation], renames: RenameMap) {
  def firrtlAnnotations: Iterable[Annotation] = annotations.flatMap(_.toFirrtl.update(renames))

}
