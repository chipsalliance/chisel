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
  val WrapOp = PrimOp("wrap")
  val ClipOp = PrimOp("clip")
  val SetBinaryPoint = PrimOp("bpset")
  val ShiftLeftBinaryPoint = PrimOp("bpshl")
  val ShiftRightBinaryPoint = PrimOp("bpshr")
  val AsClockOp = PrimOp("asClock")
}

abstract class Arg {
  def fullName(ctx: Component): String = name
  def name: String
}

case class Node(id: HasId) extends Arg {
  override def fullName(ctx: Component): String = id.getOptionRef match {
    case Some(arg) => arg.fullName(ctx)
    case None => id.suggestedName.getOrElse("??")
  }
  def name: String = id.getOptionRef match {
    case Some(arg) => arg.name
    case None => id.suggestedName.getOrElse("??")
  }
}

abstract class LitArg(val num: BigInt, widthArg: Width) extends Arg {
  private[chisel3] def forcedWidth = widthArg.known
  private[chisel3] def width: Width = if (forcedWidth) widthArg else Width(minWidth)
  override def fullName(ctx: Component): String = name
  // Ensure the node representing this LitArg has a ref to it and a literal binding.
  def bindLitArg[T <: Bits](bits: T): T = {
    bits.bind(ElementLitBinding(this))
    bits.setRef(this)
    bits
  }

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

case class SLit(n: BigInt, w: Width) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asSInt(${ULit(unsigned, width).name})"
  }
  def minWidth: Int = 1 + n.bitLength
}

case class FPLit(n: BigInt, w: Width, binaryPoint: BinaryPoint) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asFixedPoint(${ULit(unsigned, width).name}, ${binaryPoint.asInstanceOf[KnownBinaryPoint].value})"
  }
  def minWidth: Int = 1 + n.bitLength
}
//TODO (chick) How to output this as unknown width
case class IntervalLit(n: BigInt, w: Width, binaryPoint: BinaryPoint) extends LitArg(n, w) {
  def name: String = {
    val unsigned = if (n < 0) (BigInt(1) << width.get) + n else n
    s"asInterval(${ULit(unsigned, width).name}, $minWidth, $minWidth, ${binaryPoint.asInstanceOf[KnownBinaryPoint].value})"
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

sealed trait Bound
sealed trait NumericBound[T] extends Bound {
  val value: T
}
sealed case class Open[T](value: T) extends NumericBound[T]
sealed case class Closed[T](value: T) extends NumericBound[T]

sealed trait Range {
  val min: Bound
  val max: Bound
  def getWidth: Width
}

sealed trait KnownIntRange extends Range {
  val min: NumericBound[Int]
  val max: NumericBound[Int]

  require( (min, max) match {
    case (Open(low_val), Open(high_val)) => low_val < high_val - 1
    case (Closed(low_val), Open(high_val)) => low_val < high_val
    case (Open(low_val), Closed(high_val)) => low_val < high_val
    case (Closed(low_val), Closed(high_val)) => low_val <= high_val
  })
}

sealed case class KnownUIntRange(min: NumericBound[Int], max: NumericBound[Int]) extends KnownIntRange {
  require (min.value >= 0)

  def getWidth: Width = max match {
    case Open(v) => Width(BigInt(v - 1).bitLength.max(1))
    case Closed(v) => Width(BigInt(v).bitLength.max(1))
  }
}

sealed case class KnownSIntRange(min: NumericBound[Int], max: NumericBound[Int]) extends KnownIntRange {

  val maxWidth = max match {
    case Open(v) => Width(BigInt(v - 1).bitLength + 1)
    case Closed(v) => Width(BigInt(v).bitLength + 1)
  }
  val minWidth = min match {
    case Open(v) => Width(BigInt(v + 1).bitLength + 1)
    case Closed(v) => Width(BigInt(v).bitLength + 1)
  }
  def getWidth: Width = maxWidth.max(minWidth)

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

sealed class IntervalRange(
    val lowerBound: firrtlir.Bound,
    val upperBound: firrtlir.Bound,
    private[chisel3] val firrtlBinaryPoint: firrtlir.Width)
  extends firrtlir.IntervalType(lowerBound, upperBound, firrtlBinaryPoint)
  with RangeType {
  (lower, upperBound) match {
    case (firrtlir.Open(begin), firrtlir.Open(end)) =>
      if(begin >= end) throw new IllegalArgumentException(s"Invalid range with ${serialize}")
      binaryPoint match {
        case KnownBinaryPoint(bp) =>
          if(begin >= end - (BigDecimal(1) / BigDecimal(BigInt(1) << bp))) {
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

  def getPossibleValues: Seq[Double] = {
    (lower, upperBound, firrtlBinaryPoint) match {
      case (firrtlir.Open(begin), firrtlir.Open(end), firrtlir.IntWidth(bp)) =>
        (begin.doubleValue until end.doubleValue by math.pow(2, -bp.doubleValue)).toSeq.tail
      case (firrtlir.Open(begin), firrtlir.Closed(end), firrtlir.IntWidth(bp)) =>
        (begin.doubleValue to end.doubleValue by math.pow(2, -bp.doubleValue)).toSeq.tail
      case (firrtlir.Closed(begin), firrtlir.Open(end), firrtlir.IntWidth(bp)) =>
        (begin.doubleValue until end.doubleValue by math.pow(2, -bp.doubleValue)).toSeq
      case (firrtlir.Closed(begin), firrtlir.Closed(end), firrtlir.IntWidth(bp)) =>
        (begin.doubleValue to end.doubleValue by math.pow(2, -bp.doubleValue)).toSeq
      case _ =>
        throw new Exception("Bounds unknown. Cannot get possible values from IntervalRange.")
    }
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

  private def shiftLeft(bound: firrtlir.Bound, n: Int): firrtlir.Bound = {
    if(n < 1 ) {
      bound
    }
    else {
      val multiplier = BigDecimal(BigInt(1) << n)
      bound match {
        case firrtlir.Open(x) => firrtlir.Open(x * multiplier)
        case firrtlir.Closed(x) => firrtlir.Closed(x * multiplier)
        case _ => firrtlir.UnknownBound
      }
    }
  }

  private def shiftRight(bound: firrtlir.Bound, n: Int): firrtlir.Bound = {
    if(n < 1 ) {
      bound
    }
    else {
      // Because of loss of significant digits, range changes too BUT to figure that out, you need to know BP
      // TODO: (chick) -- can add bp info (if known); see firrtl primops
      /*
      val divisor = 1 << n
      bound match {
        case firrtlir.Open(x) => firrtlir.Open(x / divisor)
        case firrtlir.Closed(x) => firrtlir.Closed(x / divisor)
        case _ => bound
      }
      */
      firrtlir.UnknownBound
    }
  }

  override def <<(that: Int): IntervalRange = {
    IntervalRange(
      shiftLeft(lowerBound, that),
      shiftLeft(upperBound, that),
      binaryPoint
    )
  }

  override def >>(that: Int): IntervalRange = {
    IntervalRange(
      shiftRight(lowerBound, that),
      shiftRight(upperBound, that),
      binaryPoint
    )
  }

  override def <<(that: KnownWidth): IntervalRange = {
    // TODO: (chick) requires being able to look at min/max of two possibly known bounds; see primop
    IntervalRange(firrtlir.UnknownBound, firrtlir.UnknownBound, binaryPoint)
  }

  override def >>(that: KnownWidth): IntervalRange = {
    // Worst case range is when this is not shifted
    this
  }

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
  // These ops straight up add widths together
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
  // TODO: (chick) check -- going by the same reasoning as for W, the resultant BP should be
  // operation dependent. BUT shr, dshl should NOT change bp
  def shiftRight(that: Int): BinaryPoint = ??? //this.op(this, (a, b) => 0 max (a - that))
  def dynamicShiftLeft(that: BinaryPoint): BinaryPoint = ???
    // this.op(that, (a, b) => a + (1 << b) - 1)

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
case class WhenEnd(sourceInfo: SourceInfo, firrtlDepth: Int, hasAlt: Boolean = false) extends Command
case class AltBegin(sourceInfo: SourceInfo) extends Command
case class OtherwiseEnd(sourceInfo: SourceInfo, firrtlDepth: Int) extends Command
case class Connect(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
case class BulkConnect(sourceInfo: SourceInfo, loc1: Node, loc2: Node) extends Command
case class Attach(sourceInfo: SourceInfo, locs: Seq[Node]) extends Command
case class ConnectInit(sourceInfo: SourceInfo, loc: Node, exp: Arg) extends Command
case class Stop(sourceInfo: SourceInfo, clock: Arg, ret: Int) extends Command
case class Port(id: Data, dir: SpecifiedDirection)
case class Printf(sourceInfo: SourceInfo, clock: Arg, pable: Printable) extends Command
abstract class Component extends Arg {
  def id: BaseModule
  def name: String
  def ports: Seq[Port]
}
case class DefModule(id: UserModule, name: String, ports: Seq[Port], commands: Seq[Command]) extends Component
case class DefBlackBox(id: BaseBlackBox, name: String, ports: Seq[Port], topDir: SpecifiedDirection, params: Map[String, Param]) extends Component

case class Circuit(name: String, components: Seq[Component], annotations: Seq[ChiselAnnotation] = Seq.empty)
