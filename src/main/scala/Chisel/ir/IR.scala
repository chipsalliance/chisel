// See LICENSE for license details.

package Chisel

case class PrimOp(val name: String) {
  override def toString: String = name
}

object PrimOp {
  val AddOp = PrimOp("add")
  val AddModOp = PrimOp("addw")
  val SubOp = PrimOp("sub")
  val SubModOp = PrimOp("subw")
  val TimesOp = PrimOp("mul")
  val DivideOp = PrimOp("div")
  val ModOp = PrimOp("mod")
  val ShiftLeftOp = PrimOp("shl")
  val ShiftRightOp = PrimOp("shr")
  val DynamicShiftLeftOp = PrimOp("dshl")
  val DynamicShiftRightOp = PrimOp("dshr")
  val BitAndOp = PrimOp("and")
  val BitOrOp = PrimOp("or")
  val BitXorOp = PrimOp("xor")
  val BitNotOp = PrimOp("not")
  val ConcatOp = PrimOp("cat")
  val BitSelectOp = PrimOp("bit")
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
  private[Chisel] def forcedWidth = widthArg.known
  private[Chisel] def width: Width = if (forcedWidth) widthArg else Width(minWidth)

  protected def minWidth: Int
  if (forcedWidth) {
    require(widthArg.get >= minWidth)
  }
}

case class ILit(n: BigInt) extends Arg {
  def name: String = n.toString
}

case class ULit(n: BigInt, w: Width) extends LitArg(n, w) {
  def name: String = "UInt<" + width + ">(\"h0" + num.toString(16) + "\")"
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

case class Ref(name: String) extends Arg
case class ModuleIO(mod: Module, name: String) extends Arg {
  override def fullName(ctx: Component): String =
    if (mod eq ctx.id) name else s"${mod.getRef.name}.$name"
}
case class Slot(imm: Node, name: String) extends Arg {
  override def fullName(ctx: Component): String =
    if (imm.fullName(ctx).isEmpty) name else s"${imm.fullName(ctx)}.${name}"
}
case class Index(imm: Arg, value: Int) extends Arg {
  def name: String = s"[$value]"
  override def fullName(ctx: Component): String = s"${imm.fullName(ctx)}[$value]"
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
  override def toString: String = "?"
}

sealed case class KnownWidth(value: Int) extends Width {
  require(value >= 0)
  def known: Boolean = true
  def get: Int = value
  def op(that: Width, f: (W, W) => W): Width = that match {
    case KnownWidth(x) => KnownWidth(f(value, x))
    case _ => that
  }
  override def toString: String = value.toString
}

abstract class Command
abstract class Definition extends Command {
  def id: HasId
  def name: String = id.getRef.name
}
case class DefPrim[T <: Data](id: T, op: PrimOp, args: Arg*) extends Definition
case class DefWire(id: Data) extends Definition
case class DefRegister(id: Data, clock: Arg, reset: Arg) extends Definition
case class DefMemory(id: HasId, t: Data, size: Int, clock: Arg) extends Definition
case class DefSeqMemory(id: HasId, t: Data, size: Int, clock: Arg) extends Definition
case class DefAccessor[T <: Data](id: T, source: Node, direction: Direction, index: Arg) extends Definition
case class DefInstance(id: Module, ports: Seq[Port]) extends Definition
case class DefPoison[T <: Data](id: T) extends Definition
case class WhenBegin(pred: Arg) extends Command
case class WhenElse() extends Command
case class WhenEnd() extends Command
case class Connect(loc: Node, exp: Arg) extends Command
case class BulkConnect(loc1: Node, loc2: Node) extends Command
case class ConnectInit(loc: Node, exp: Arg) extends Command
case class Component(id: Module, name: String, ports: Seq[Port], commands: Seq[Command]) extends Arg
case class Port(id: Data, dir: Direction)

case class Circuit(name: String, components: Seq[Component], refMap: RefMap) {
  def emit: String = new Emitter(this).toString
}
