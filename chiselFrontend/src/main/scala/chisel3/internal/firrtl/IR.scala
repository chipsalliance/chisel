// See LICENSE for license details.

package chisel3.internal.firrtl

import chisel3._
import core._
import chisel3.internal._
import chisel3.internal.sourceinfo.{NoSourceInfo, SourceInfo}
import _root_.firrtl.annotations.Annotation
import _root_.firrtl.{ir => firrtlir}
import _root_.firrtl.passes.{IsAdd, IsMax}
import _root_.firrtl.PrimOps

case class PrimOp(val name: String) {
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
  val XorReduceOp = PrimOp("xorr")
  val ConvertOp = PrimOp("cvt")
  val AsUIntOp = PrimOp("asUInt")
  val AsSIntOp = PrimOp("asSInt")
  val AsFixedPointOp = PrimOp("asFixedPoint")
  val AsIntervalOp = PrimOp("asInterval")
  val SetBinaryPoint = PrimOp("bpset")
  val AsClockOp = PrimOp("asClock")
}

abstract class Arg {
  def fullName(ctx: Component): String = name
  def name: String
}

case class Node(id: HasId) extends Arg {
  override def fullName(ctx: Component): String = id.getRef.fullName(ctx)
  def name: String = id.getRef.name
}

abstract class LitArg(val num: BigInt, widthArg: Width) extends Arg {
  private[chisel3] def forcedWidth = widthArg.known
  private[chisel3] def width: Width = if (forcedWidth) widthArg else Width(minWidth)

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

  require(n >= 0, s"UInt literal ${n} is negative")
}

case class SLit(n: BigInt, w: Width, directUse: Boolean = true) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asSInt(${ULit(unsigned, width).name})"
  }
  // TODO: (chick) check my logic please
  // If SLit is directly created by the Chisel user, to account for #
  // of bits needed to represent a negative n value, you need the + 1
  // HOWEVER, if SLit is applied via IntervalLit or FPLit, n has already
  // been converted two UNSIGNED, so no need for + 1
  def minWidth: Int = 
    if (directUse) 1 + n.bitLength
    else n.bitLength
}

case class FPLit(n: BigInt, w: Width, binaryPoint: BinaryPoint) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asFixedPoint(${SLit(unsigned, width, directUse = false).name}, ${binaryPoint.asInstanceOf[KnownBinaryPoint].value})"
  }
  def minWidth: Int = 1 + n.bitLength
}

// TODO: (chick) double check?; if binaryPoint > 0, n has already been shifted up 
case class IntervalLit(n: BigInt, w: Width, binaryPoint: BinaryPoint) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
//    s"asInterval(${SLit(unsigned, width).name}, ${binaryPoint.asInstanceOf[KnownBinaryPoint].value})"
    binaryPoint match {
      case KnownBinaryPoint(bp) =>
        s"asInterval(${SLit(unsigned, width, directUse = false).name}, $n, $n, $bp)"
      case _ =>
        throw new Exception("Interval Lit requires known binary point")
    }
    
  }
  val range: IntervalRange = {
    new IntervalRange(IntervalRange.getBound(isClosed = true, BigDecimal(n)),
      IntervalRange.getBound(isClosed = true, BigDecimal(n)), IntervalRange.getRangeWidth(binaryPoint))
  }
  def minWidth: Int = 1 + n.bitLength
}

case class Ref(name: String) extends Arg
case class ModuleIO(mod: BaseModule, name: String) extends Arg {
  override def fullName(ctx: Component): String =
    if (mod eq ctx.id) name else s"${mod.getRef.name}.$name"
}
case class Slot(imm: Node, name: String) extends Arg {
  override def fullName(ctx: Component): String =
    if (imm.fullName(ctx).isEmpty) name else s"${imm.fullName(ctx)}.${name}"
}
case class Index(imm: Arg, value: Arg) extends Arg {
  def name: String = s"[$value]"
  override def fullName(ctx: Component): String = s"${imm.fullName(ctx)}[${value.fullName(ctx)}]"
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

//TODO: chick: extract logic, delete all this
//case object UnknownRange extends Range {
//  val min = UnknownBound
//  val max = UnknownBound
//
//  def * (that: Range): Range = this
//  def +& (that: Range): Range = this
//  def -& (that: Range): Range = this
//  def << (that: Int): Range = this
//  def >> (that: Int): Range = this
//  def << (that: KnownWidth): Range = this
//  def >> (that: KnownWidth): Range = this
//
//  override def getWidth: Width = ???
//  override def toString: String = ""
//  def merge(that: Range): Range = UnknownRange
//}
//
//sealed trait KnownBigIntRange extends Range {
//  val min: NumericBound[BigInt]
//  val max: NumericBound[BigInt]
//
//  require( (min, max) match {
//    case (Open(low_val), Open(high_val)) => low_val < high_val - 1
//    case (Closed(low_val), Open(high_val)) => low_val < high_val
//    case (Open(low_val), Closed(high_val)) => low_val < high_val
//    case (Closed(low_val), Closed(high_val)) => low_val <= high_val
//  })
//  override def toString: String = {
//    (min, max) match {
//      case (Open(low_val), Open(high_val)) => s"($min, $max)"
//      case (Closed(low_val), Open(high_val)) => s"[$min, $max"
//      case (Open(low_val), Closed(high_val)) => s"$min, $max"
//      case (Closed(low_val), Closed(high_val)) => s"$min, $max"
//    }
//    s"$min, $max"
//  }
//}
//
//sealed case class KnownIntervalRange(
//                                      min: NumericBound[BigInt],
//                                      max: NumericBound[BigInt],
//                                      binaryPoint: BinaryPoint = KnownBinaryPoint(0))
//  extends KnownBigIntRange{
//  val maxWidth = max match {
//    case Open(v) => Width((v - 1).bitLength + 1)
//    case Closed(v) => Width(v.bitLength + 1)
//  }
//  val minWidth = min match {
//    case Open(v) => Width((v + 1).bitLength + 1)
//    case Closed(v) => Width(v.bitLength + 1)
//  }
//
//  def getWidth: Width = maxWidth.max(minWidth)
//
//  override def toString: String = {
//    (min, max) match {
//      case (Open(low_val), Open(high_val)) => s"[${min.value + 1}, ${max.value - 1})]"
//      case (Closed(low_val), Open(high_val)) => s"[[${min.value} ${max.value - 1}]"
//      case (Open(low_val), Closed(high_val)) => s"[${min.value + 1} ${max .value}]"
//      case (Closed(low_val), Closed(high_val)) => s"[${min.value} ${max .value}]"
//    }
//  }
//  def getValues: (BigInt, BigInt) = (min, max) match {
//    case (Open(low_val), Open(high_val)) => (min.value + 1, max.value - 1)
//    case (Closed(low_val), Open(high_val)) => (min.value, max.value - 1)
//    case (Open(low_val), Closed(high_val)) => (min.value + 1, max.value)
//    case (Closed(low_val), Closed(high_val)) => (min.value, max.value)
//  }
//  def getValues(that: Range): Option[(BigInt, BigInt, BinaryPoint)] = that match {
//    case KnownIntervalRange(Open(l), Open(h), bP) => Some((l + 1, h - 1, bP))
//    case KnownIntervalRange(Closed(l), Open(h), bP) => Some((l, h - 1, bP))
//    case KnownIntervalRange(Open(l), Closed(h), bP) => Some((l + 1, h, bP))
//    case KnownIntervalRange(Closed(l), Closed(h), bP) => Some((l, h, bP))
//    case _ => None
//  }
//  private val (low, high) = getValues
//  def * (that: Range): Range = getValues(that) match {
//    case Some((l, h, bp)) => KnownIntervalRange(Closed(low * l), Closed(high * h), binaryPoint max bp)
//    case _ => UnknownRange
//  }
//  def +& (that: Range): Range = getValues(that) match {
//    case Some((l, h, bp)) => KnownIntervalRange(Closed(low + l), Closed(high + h), binaryPoint max bp)
//    case _ => UnknownRange
//  }
//  def -& (that: Range): Range = getValues(that) match {
//    case Some((l, h, bp)) => KnownIntervalRange(Closed(low - h), Closed(high - l), binaryPoint max bp)
//    case _ => UnknownRange
//  }
//  def << (that: Int): Range = KnownIntervalRange(Closed(low << that), Closed(high << that), binaryPoint)
//  def >> (that: Int): Range = KnownIntervalRange(Closed(low >> that), Closed(high >> that), binaryPoint)
//  def << (width: KnownWidth): Range = KnownIntervalRange(Closed(low), Closed(high << (BigInt(2) << width.value).toInt), binaryPoint)
//  def >> (width: KnownWidth): Range = KnownIntervalRange(Closed(low >> (BigInt(2) << width.value).toInt), Closed(high), binaryPoint)
//  def merge(that: Range): Range = that match {
//    case UnknownRange => UnknownRange
//    case KnownIntervalRange(lo, hi, bp) =>
//      val mergeLow = (min, lo) match {
//        case (Open(a), Open(b))      if a <= b => Open(a)
//        case (Open(a), Open(b))      if b < a  => Open(b)
//        case (Closed(a), Open(b))    if a <= b => Closed(a)
//        case (Closed(a), Open(b))    if b < a  => Open(b)
//        case (Open(a), Closed(b))    if a < b  => Open(a)
//        case (Open(a), Closed(b))    if b <= a => Closed(b)
//        case (Closed(a), Closed(b))  if a <= b => Closed(a)
//        case (Closed(a), Closed(b))  if b < a  => Closed(b)
//      }
//      val mergeHigh = (max, hi) match {
//        case (Open(a), Open(b))      if a >= b => Open(a)
//        case (Open(a), Open(b))      if b < a  => Open(b)
//        case (Closed(a), Open(b))    if a >= b => Closed(a)
//        case (Closed(a), Open(b))    if b > a  => Open(b)
//        case (Open(a), Closed(b))    if a > b  => Open(a)
//        case (Open(a), Closed(b))    if b >= a => Closed(b)
//        case (Closed(a), Closed(b))  if a >= b => Closed(a)
//        case (Closed(a), Closed(b))  if b > a  => Closed(b)
//      }
//      KnownIntervalRange(mergeLow, mergeHigh, bp max binaryPoint)
//  }
//}

sealed class IntervalRange(
    lowerBound: firrtlir.Bound,
    upperBound: firrtlir.Bound,
    private[chisel3] val firrtlBinaryPoint: firrtlir.Width)
  extends firrtlir.IntervalType(lowerBound, upperBound, firrtlBinaryPoint)
  with RangeType {
  (lower, upperBound) match {
    case (firrtlir.Open(begin), firrtlir.Open(end)) =>
      if(begin >= end) throw new IllegalArgumentException(s"Invalid range with ${serialize}")
      binaryPoint match {
        case KnownBinaryPoint(bp) =>
          if(begin >= end - (BigDecimal(1) / (1 << bp))) {
            throw new IllegalArgumentException(s"Invalid range with ${serialize}")
          }
        case _ =>
      }
    case (firrtlir.Open(begin), firrtlir.Closed(end)) =>
      if(begin >= end) throw new IllegalArgumentException(s"Invalid range with ${serialize}")
    case (firrtlir.Closed(begin), firrtlir.Open(end)) =>
      if(begin >= end) throw new IllegalArgumentException(s"Invalid range with ${serialize}")
    case (firrtlir.Closed(begin), firrtlir.Closed(end)) =>
      if(begin > end) throw new IllegalArgumentException(s"Invalid range with ${serialize}")
    case _ =>
  }
  override def getWidth: Width = {
    width match {
      case firrtlir.IntWidth(n) => KnownWidth(n.toInt)
      case firrtlir.UnknownWidth => UnknownWidth()
    }
  }

  // TODO: (chick) How to implement properly? (Angie)
  override def *(that: IntervalRange): IntervalRange = {
    IntervalRange(firrtlir.UnknownBound, firrtlir.UnknownBound, firrtlir.UnknownWidth)
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

  private def doFirrtlOp(op: firrtlir.PrimOp, that: Int): IntervalRange = {
    PrimOps.set_primop_type(
      firrtlir.DoPrim(op,
        Seq(firrtlir.Reference("a", this)), Seq(BigInt(that)), firrtlir.UnknownType)
    ).tpe match {
      case i: firrtlir.IntervalType => IntervalRange(i.lower, i.upper, i.point)
      case other => sys.error("BAD!")
    }
  }

  override def +&(that: IntervalRange): IntervalRange = {
//    doFirrtlOp(PrimOps.Add, that)
    IntervalRange(firrtlir.UnknownBound, firrtlir.UnknownBound, firrtlir.UnknownWidth)
  }
  private def getRange(op: firrtlir.PrimOp, tpe1: IntervalRange, tpe2: IntervalRange): IntervalRange = ???
  override def -&(that: IntervalRange): IntervalRange = {
//    doFirrtlOp(PrimOps.Sub, that)
    IntervalRange(firrtlir.UnknownBound, firrtlir.UnknownBound, firrtlir.UnknownWidth)
  }

  override def <<(that: Int): IntervalRange = ???

  override def >>(that: Int): IntervalRange = ???

  override def <<(that: KnownWidth): IntervalRange = ???

  override def >>(that: KnownWidth): IntervalRange = ???

  override def merge(that: IntervalRange): IntervalRange = ???

  def binaryPoint: BinaryPoint = {
    firrtlBinaryPoint match {
      case firrtlir.IntWidth(n) =>
        core.assert(n < Int.MaxValue, s"binary point value $n is out of range")
        KnownBinaryPoint(n.toInt)
      case _ => UnknownBinaryPoint
    }
  }
}

object IntervalRange {
  def apply(lower: firrtlir.Bound, upper: firrtlir.Bound, firrtlBinaryPoint: firrtlir.Width): IntervalRange = {
    new IntervalRange(lower, upper, firrtlBinaryPoint)
  }

  def apply(lower: firrtlir.Bound, upper: firrtlir.Bound, binaryPoint: BinaryPoint): IntervalRange = {
    new IntervalRange(lower, upper, IntervalRange.getBinaryPoint(binaryPoint))
  }

  def apply(lower: firrtlir.Bound, upper: firrtlir.Bound, binaryPoint: Int): IntervalRange = {
    IntervalRange(lower, upper, BinaryPoint(binaryPoint))
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

  def unknownRange: IntervalRange = new IntervalRange(firrtlir.UnknownBound, firrtlir.UnknownBound, firrtlir.UnknownWidth)
}


object Width {
  def apply(x: Int): Width = KnownWidth(x)
  def apply(): Width = UnknownWidth()
}

sealed abstract class Width {
  type W = Int
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
case class DefMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: Int) extends Definition
case class DefSeqMemory(sourceInfo: SourceInfo, id: HasId, t: Data, size: Int) extends Definition
case class DefMemPort[T <: Data](sourceInfo: SourceInfo, id: T, source: Node, dir: MemPortDirection, index: Arg, clock: Arg) extends Definition
case class DefInstance(sourceInfo: SourceInfo, id: BaseModule, ports: Seq[Port]) extends Definition
case class WhenBegin(sourceInfo: SourceInfo, pred: Arg) extends Command
case class WhenEnd(sourceInfo: SourceInfo) extends Command
case class Connect(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
case class BulkConnect(sourceInfo: SourceInfo, loc1: Node, loc2: Node) extends Command
case class Attach(sourceInfo: SourceInfo, locs: Seq[Node]) extends Command
case class ConnectInit(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
case class Stop(sourceInfo: SourceInfo, clock: Arg, ret: Int) extends Command
case class Port(id: Data, dir: Direction)
case class Printf(sourceInfo: SourceInfo, clock: Arg, pable: Printable) extends Command
abstract class Component extends Arg {
  def id: BaseModule
  def name: String
  def ports: Seq[Port]
}
case class DefModule(id: UserModule, name: String, ports: Seq[Port], commands: Seq[Command]) extends Component
case class DefBlackBox(id: BaseBlackBox, name: String, ports: Seq[Port], params: Map[String, Param]) extends Component

case class Circuit(name: String, components: Seq[Component], annotations: Seq[Annotation] = Seq.empty)
