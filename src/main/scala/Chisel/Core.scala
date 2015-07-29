package Chisel
import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, Stack, HashSet, HashMap, LinkedHashMap}
import java.lang.reflect.Modifier._
import java.lang.Double.longBitsToDouble
import java.lang.Float.intBitsToFloat

class GenSym {
  private var counter = -1
  def nextInt: Int = {
    counter += 1
    counter
  }
  def next(name: String): String =
    name + "_" + nextInt
}

object Builder {
  val components = new ArrayBuffer[Component]()
  val genSym = new GenSym()
  val switchKeyz = new Stack[Stack[Bits]]()
  def switchKeys = switchKeyz.top
  def pushScope = {
    switchKeyz.push(new Stack[Bits]())
  }
  def popScope = {
    switchKeyz.pop()
  }
  val modules = new HashMap[String,Module]()
  def addModule(mod: Module) {
    modules(mod.cid) = mod
  }
  val modulez = new Stack[Module]()
  def pushModule(mod: Module) {
    modulez.push(mod)
  }
  def getComponent(): Module = if (modulez.length > 0) modulez.head else null
  def popModule() {
    modulez.pop
  }
  val componentNames = new HashSet[String]()
  def UniqueComponent(name: String, ports: Seq[Port], body: Command) = {
    val res = Component(if (componentNames.contains(name)) genSym.next(name) else name, ports, body)
    componentNames += name
    res
  }
  val commandz = new Stack[ArrayBuffer[Command]]()
  def commands = commandz.top
  def pushCommand(cmd: Command) = commands += cmd
  def commandify(cmds: ArrayBuffer[Command]): Command = {
    if (cmds.length == 0)
      EmptyCommand()
    else if (cmds.length == 1)
      cmds(0)
    else
      Begin(cmds.toList)
  }
  def pushCommands = 
    commandz.push(new ArrayBuffer[Command]())
  def popCommands: Command = {
    val newCommands = commands
    commandz.pop()
    commandify(newCommands)
  }
  def collectCommands[T <: Module](f: => T): (Command, T) = {
    pushCommands
    val mod = f
    // mod.setRefs
    (popCommands, mod)
  }

  private val refmap = new HashMap[String,Immediate]()

  def legalizeName (name: String) = {
    if (name == "mem" || name == "node" || name == "wire" ||
        name == "reg" || name == "inst")
      // genSym.next(name)
      name + "__"
    else
      name
  }

  def setRefForId(id: String, name: String, overwrite: Boolean = false) {
    if (overwrite || !refmap.contains(id)) {
      refmap(id) = Ref(legalizeName(name))
    }
  }

  def setFieldForId(parentid: String, id: String, name: String) {
    refmap(id) = Slot(Alias(parentid), legalizeName(name))
  }

  def setIndexForId(parentid: String, id: String, index: Int) {
    refmap(id) = Index(Alias(parentid), index)
  }

  def getRefForId(id: String): Immediate = {
    if (refmap.contains(id)) {
      refmap(id)
    } else  {
      val ref = Ref(genSym.next("T"))
      refmap(id) = ref
      ref
    }
  }

  def build[T <: Module](f: => T): (Circuit, T) = {
    val (cmd, mod) = collectCommands(f)
    setRefForId(mod.cid, mod.name)
    (Circuit(components, components.last.name), mod)
  }

}

import Builder._

/// CHISEL IR

case class PrimOp(val name: String) {
  override def toString = name
}

object PrimOp {
  val AddOp = PrimOp("add")
  val AddModOp = PrimOp("add-wrap")
  val SubOp = PrimOp("sub")
  val SubModOp = PrimOp("sub-wrap")
  val TimesOp = PrimOp("mul")
  val DivideOp = PrimOp("div")
  val ModOp = PrimOp("mod")
  val ShiftLeftOp = PrimOp("shl")
  val ShiftRightOp = PrimOp("shr")
  val DynamicShiftLeftOp = PrimOp("dshl")
  val DynamicShiftRightOp = PrimOp("dshr")
  val BitAndOp = PrimOp("bit-and")
  val BitOrOp = PrimOp("bit-or")
  val BitXorOp = PrimOp("bit-xor")
  val BitNotOp = PrimOp("bit-not")
  val ConcatOp = PrimOp("cat")
  val BitSelectOp = PrimOp("bit")
  val BitsExtractOp = PrimOp("bits")
  val LessOp = PrimOp("lt")
  val LessEqOp = PrimOp("leq")
  val GreaterOp = PrimOp("gt")
  val GreaterEqOp = PrimOp("geq")
  val EqualOp = PrimOp("eq")
  val PatternEqualOp = PrimOp("pattern-equal")
  val PadOp = PrimOp("pad")
  val NotEqualOp = PrimOp("neq")
  val NegOp = PrimOp("neg")
  val MultiplexOp = PrimOp("mux")
  val XorReduceOp = PrimOp("xorr")
  val ConvertOp = PrimOp("convert")
  val AsUIntOp = PrimOp("as-UInt")
  val AsSIntOp = PrimOp("as-SInt")
}
import PrimOp._

abstract class Immediate {
  def fullname: String
  def name: String
  def debugName = fullname
}

abstract class Arg extends Immediate {
  def fullname: String
  def name: String
}

case class Alias(val id: String) extends Arg {
  def fullname = getRefForId(id).fullname
  def name = getRefForId(id).name
  override def debugName = getRefForId(id).debugName
  def emit: String = "Alias(" + id + ")"
}

abstract class LitArg (val num: BigInt, val width: Int) extends Arg {
}

case class ULit(n: BigInt, w: Int = -1) extends LitArg(n, w) {
  def fullname = name
  def name = "UInt<" + width + ">(" + num + ")"
}

case class SLit(n: BigInt, w: Int = -1) extends LitArg(n, w) {
  def fullname = name
  def name = "SInt<" + width + ">(" + num + ")"
}

case class Ref(val name: String) extends Immediate {
  def fullname = name
}
case class Slot(val imm: Immediate, val name: String) extends Immediate {
  def fullname = {
    val imm_fullname = imm.fullname
    if (imm_fullname == "this") name else imm_fullname + "." + name
  }
  override def debugName = {
    val imm_debugName = imm.debugName
    if (imm_debugName == "this") name else imm_debugName + "." + name
  }
}
case class Index(val imm: Immediate, val value: Int) extends Immediate {
  def name = "[" + value + "]"
  def fullname = imm.fullname + "[" + value + "]"
  override def debugName = imm.debugName + "." + value
}

case class Port(val id: String, val dir: Direction, val kind: Kind);

abstract class Width;
case class UnknownWidth() extends Width;
case class IntWidth(val value: Int) extends Width;

abstract class Kind(val isFlip: Boolean);
case class UnknownType(flip: Boolean) extends Kind(flip);
case class UIntType(val width: Width, flip: Boolean) extends Kind(flip);
case class SIntType(val width: Width, flip: Boolean) extends Kind(flip);
case class FloType(flip: Boolean) extends Kind(flip);
case class DblType(flip: Boolean) extends Kind(flip);
case class BundleType(val ports: Seq[Port], flip: Boolean) extends Kind(flip);
case class VectorType(val size: Int, val kind: Kind, flip: Boolean) extends Kind(flip);

abstract class Command;
abstract class Definition extends Command {
  def id: String
  def name = getRefForId(id).name
}
case class DefUInt(val id: String, val value: BigInt, val width: Int) extends Definition;
case class DefSInt(val id: String, val value: BigInt, val width: Int) extends Definition;
case class DefFlo(val id: String, val value: Float) extends Definition;
case class DefDbl(val id: String, val value: Double) extends Definition;
case class DefPrim(val id: String, val kind: Kind, val op: PrimOp, val args: Seq[Arg], val lits: Seq[BigInt]) extends Definition;
case class DefWire(val id: String, val kind: Kind) extends Definition;
case class DefRegister(val id: String, val kind: Kind) extends Definition;
case class DefMemory(val id: String, val kind: Kind, val size: Int) extends Definition;
case class DefSeqMemory(val id: String, val kind: Kind, val size: Int) extends Definition;
case class DefAccessor(val id: String, val source: Alias, val direction: Direction, val index: Arg) extends Definition;
case class DefInstance(val id: String, val module: String) extends Definition;
case class Conditionally(val prep: Command, val pred: Arg, val conseq: Command, var alt: Command) extends Command;
case class Begin(val body: List[Command]) extends Command();
case class Connect(val loc: Alias, val exp: Arg) extends Command;
case class BulkConnect(val loc1: Alias, val loc2: Alias) extends Command;
case class ConnectInit(val loc: Alias, val exp: Arg) extends Command;
case class ConnectInitIndex(val loc: Alias, val index: Int, val exp: Arg) extends Command;
case class EmptyCommand() extends Command;

case class Component(val name: String, val ports: Seq[Port], val body: Command);
case class Circuit(val components: Seq[Component], val main: String);

object Commands {
  val NoLits = Seq[BigInt]()
}

import Commands._

/// COMPONENTS

sealed abstract class Direction(val name: String) {
  override def toString = name
}
object INPUT  extends Direction("input")
object OUTPUT extends Direction("output")
object NO_DIR extends Direction("?")

object Direction {
  def flipDirection(dir: Direction) = {
    dir match {
      case INPUT => OUTPUT
      case OUTPUT => INPUT
      case NO_DIR => NO_DIR
    }
  }
}
import Direction._

/// CHISEL FRONT-END

abstract class Id {
  protected[Chisel] val _id = genSym.nextInt
  protected[Chisel] val cid = "id_" + _id

  def defd: this.type = this
}

object debug {
  // TODO:
  def apply (arg: Data) = arg
}

abstract class Data(dirArg: Direction) extends Id {
  private[Chisel] val mod = getComponent()
  mod._nodes += this

  def toType: Kind
  var isFlipVar = dirArg == INPUT
  def isFlip = isFlipVar
  def dir: Direction = if (isFlip) INPUT else OUTPUT
  def setDir(dir: Direction) {
    isFlipVar = (dir == INPUT)
  }
  def asInput: this.type = {
    setDir(INPUT)
    this
  }
  def asOutput: this.type = {
    setDir(OUTPUT)
    this
  }
  def flip(): this.type = {
    isFlipVar = !isFlipVar
    this
  }
  def :=(other: Data) = 
    pushCommand(Connect(this.lref, other.ref))
  def <>(other: Data) = 
    pushCommand(BulkConnect(this.lref, other.lref))
  private[Chisel] def collectElts = { }
  def cloneType: this.type
  def cloneTypeWidth(width: Int): this.type
  def lref: Alias = 
    Alias(cid)
  def ref: Arg = 
    if (isLit) litArg() else Alias(cid)
  def name = getRefForId(cid).name
  def debugName = mod.debugName + "." + getRefForId(cid).debugName
  def litArg(): LitArg = null
  def litValue(): BigInt = None.get
  def isLit(): Boolean = false
  def floLitValue: Float = intBitsToFloat(litValue().toInt)
  def dblLitValue: Double = longBitsToDouble(litValue().toLong)
  def getWidth: Int = flatten.map(_.getWidth).reduce(_ + _)
  def maxWidth(other: Data, amt: BigInt): Int = -1
  def sumWidth(amt: BigInt): Int = -1
  def sumWidth(other: Data, amt: BigInt): Int = -1
  def flatten: IndexedSeq[Bits]
  def fromBits(n: Bits): this.type = {
    var i = 0
    val wire = Wire(this.cloneType)
    for (x <- wire.flatten.reverse) {
      x := n(i + x.getWidth-1, i)
      i += x.getWidth
    }
    wire.asInstanceOf[this.type]
  }
  def toBits(): UInt = {
    val elts = this.flatten.reverse
    Cat(elts.head, elts.tail:_*).asUInt
  }
  def makeLit(value: BigInt, width: Int): this.type =
    this.fromBits(Bits(value, width))

  def toPort: Port = Port(cid, dir, toType)
  var isReg_ = false
  def isReg = isReg_
  def params = if(Driver.parStack.isEmpty) Parameters.empty else Driver.parStack.top
}

object Wire {
  def apply[T <: Data](t: T = null, init: T = null): T = {
    val mType = if (t == null) init else t
    if(mType == null) 
      throw new Exception("cannot infer type of Init.")
    val x = mType.cloneType
    // TODO: COME UP WITH MORE ROBUST WAY TO HANDLE THIS
    pushCommand(DefWire(x.defd.cid, x.toType))
    if (init != null) 
      pushCommand(Connect(x.lref, init.ref))
    x
  }
}

object Reg {
  def apply[T <: Data](t: T = null, next: T = null, init: T = null): T = {
    var mType = t
    if(mType == null) 
      mType = next
    if(mType == null) 
      mType = init
    if(mType == null) 
      throw new Exception("cannot infer type of Reg.")
    val x = mType.cloneType
    x.isReg_ = true
    pushCommand(DefRegister(x.defd.cid, x.toType))
    if (init != null) 
      pushCommand(ConnectInit(x.lref, init.ref))
    if (next != null) 
      x := next
    x
  }
  def apply[T <: Data](outType: T): T = Reg[T](outType, null.asInstanceOf[T], null.asInstanceOf[T])
}

object Mem {
  def apply[T <: Data](t: T, size: Int): Mem[T] = {
    val mt  = t.cloneType
    val mem = new Mem(mt, size)
    pushCommand(DefMemory(mt.defd.cid, mt.toType, size))
    mem
  }
}

class Mem[T <: Data](protected[Chisel] val t: T, n: Int) extends VecLike[T] {
  def length: Int = n
  def apply(idx: Int): T = apply(UInt(idx))
  def apply(idx: UInt): T = {
    val x = t.cloneType
    pushCommand(DefAccessor(x.defd.cid, Alias(t.cid), NO_DIR, idx.ref))
    x
  }

  def read(idx: UInt): T = apply(idx)
  def write(idx: UInt, data: T): Unit = apply(idx) := data
  def write(idx: UInt, data: T, mask: T): Unit = {
    // This is totally fucked, but there's no true write mask support yet
    val mask1 = mask.toBits
    write(idx, t.fromBits((read(idx).toBits & ~mask1) | (data.toBits & mask1)))
  }

  def name = getRefForId(t.cid).name
  def debugName = t.mod.debugName + "." + getRefForId(t.cid).debugName
}

object SeqMem {
  def apply[T <: Data](t: T, size: Int): SeqMem[T] =
    new SeqMem(t, size)
}

// For now, implement SeqMem in terms of Mem
class SeqMem[T <: Data](t: T, n: Int) {
  private val mem = Mem(t, n)

  def read(addr: UInt): T = mem.read(Reg(next = addr))
  def read(addr: UInt, enable: Bool): T = mem.read(RegEnable(addr, enable))

  def write(addr: UInt, data: T): Unit = mem.write(addr, data)
  def write(addr: UInt, data: T, mask: T): Unit = mem.write(addr, data, mask)
}

object Vec {
  def apply[T <: Data](gen: T, n: Int): Vec[T] = 
    new Vec((0 until n).map(i => gen.cloneType))
  def apply[T <: Data](elts: Iterable[T]): Vec[T] = {
    val vec = new Vec[T](elts.map(e => elts.head.cloneType))
    if (vec.isReg)
      throw new Exception("Vec of Reg Deprecated.")
    pushCommand(DefWire(vec.defd.cid, vec.toType))

    for ((v, e) <- vec zip elts)
      v := e
    vec
  }
  def apply[T <: Data](elt0: T, elts: T*): Vec[T] =
    apply(elt0 +: elts.toSeq)
  def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] = 
    apply((0 until n).map(i => gen(i)))
  def fill[T <: Data](n: Int)(gen: => T): Vec[T] = 
    Vec.tabulate(n){ i => gen }
}

abstract class Aggregate(dirArg: Direction) extends Data(dirArg) {
  def cloneTypeWidth(width: Int): this.type = cloneType
}

class Vec[T <: Data](elts: Iterable[T], dirArg: Direction = NO_DIR) extends Aggregate(dirArg) with VecLike[T] {
  private val self = elts.toIndexedSeq
  private lazy val elt0 = elts.head

  override def collectElts: Unit =
    for ((e, i) <- self zipWithIndex)
      setIndexForId(cid, e.cid, i)

  def <> (that: Iterable[T]): Unit =
    for ((a, b) <- this zip that)
      a <> b

  def := (that: Iterable[T]): Unit =
    for ((a, b) <- this zip that)
      a := b

  override def isFlip = isFlipVar ^ (!elts.isEmpty && elt0.isFlip)

  def apply(idx: UInt): T = {
    val x = elt0.cloneType
    pushCommand(DefAccessor(x.defd.cid, Alias(cid), NO_DIR, idx.ref))
    x
  }
  def apply(idx: Int): T = 
    self(idx)
  def toPorts: Seq[Port] =
    self.map(d => d.toPort)
  def toType: Kind = {
    val eltType = if (elts.isEmpty) UIntType(UnknownWidth(), isFlipVar) else elt0.toType
    VectorType(self.size, eltType, isFlipVar)
  }
  override def cloneType: this.type =
    Vec(elt0.cloneType, self.size).asInstanceOf[this.type]
  override def flatten: IndexedSeq[Bits] =
    self.map(_.flatten).reduce(_ ++ _)

  def length: Int = self.size

  def read(idx: UInt): T = apply(idx)
  def write(idx: UInt, data: T): Unit = apply(idx) := data
}

trait VecLike[T <: Data] extends collection.IndexedSeq[T] {
  def read(idx: UInt): T
  def write(idx: UInt, data: T): Unit
  def apply(idx: UInt): T

  def forall(p: T => Bool): Bool = (this map p).fold(Bool(true))(_&&_)
  def exists(p: T => Bool): Bool = (this map p).fold(Bool(false))(_||_)
  def contains(x: T) (implicit evidence: T <:< Bits): Bool = this.exists(_ === x)
  def count(p: T => Bool): UInt = PopCount((this map p).toSeq)

  private def indexWhereHelper(p: T => Bool) = this map p zip (0 until length).map(i => UInt(i))
  def indexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p))
  def lastIndexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p).reverse)
  def onlyIndexWhere(p: T => Bool): UInt = Mux1H(indexWhereHelper(p))
}

// object chiselCast {
//   def apply[S <: Data, T <: Bits](i: S)(gen: Int => T): T = {
//     val b = i.toBits
//     val x = gen(b.getWidth)
//     pushCommand(DefPrim(x.defd.id, x.toType, ConvertOp, Seq(b.ref), NoLits))
//     x
//   }
// }

object BitPat {
  private def parse(x: String): (BigInt, BigInt, Int) = {
    require(x.head == 'b', "BINARY BitPats ONLY")
    var bits = BigInt(0)
    var mask = BigInt(0)
    for (d <- x.tail) {
      if (d != '_') {
        if (!"01?".contains(d)) ChiselError.error({"Literal: " + x + " contains illegal character: " + d})
        mask = (mask << 1) + (if (d == '?') 0 else 1)
        bits = (bits << 1) + (if (d == '1') 1 else 0)
      }
    }
    (bits, mask, x.length-1)
  }

  def apply(n: String): BitPat = {
    val (bits, mask, width) = parse(n)
    new BitPat(bits, mask, width)
  }

  def DC(width: Int): BitPat = BitPat("b" + ("?" * width))

  // BitPat <-> UInt
  implicit def BitPatToUInt(x: BitPat): UInt = {
    require(x.mask == (BigInt(1) << x.getWidth)-1)
    UInt(x.value, x.getWidth)
  }
  implicit def apply(x: UInt): BitPat = {
    require(x.isLit)
    BitPat("b" + x.litValue.toString(2))
  }
}

class BitPat(val value: BigInt, val mask: BigInt, width: Int) {
  def getWidth: Int = width
  def === (other: Bits): Bool = UInt(value) === (other & UInt(mask))
  def != (other: Bits): Bool = !(this === other)
}

abstract class Element(dirArg: Direction, val width: Int) extends Data(dirArg) {
  override def getWidth: Int = width
}

abstract class Bits(dirArg: Direction, width: Int, lit: Option[LitArg]) extends Element(dirArg, width) {
  override def litArg(): LitArg = lit.get
  override def isLit(): Boolean = lit.isDefined
  override def litValue(): BigInt = lit.get.num
  override def cloneType : this.type = cloneTypeWidth(width)
  def fromInt(x: BigInt): this.type = makeLit(x, -1)

  override def flatten: IndexedSeq[Bits] = IndexedSeq(this)

  final def apply(x: BigInt): Bool = {
    if (isLit()) Bool((litValue() >> x.toInt) & 1)
    else {
      val d = Bool()
      pushCommand(DefPrim(d.defd.cid, d.toType, BitSelectOp, Seq(this.ref), Seq(x)))
      d
    }
  }
  final def apply(x: Int): Bool =
    apply(BigInt(x))
  final def apply(x: UInt): Bool =
    (this >> x)(0)

  final def apply(x: BigInt, y: BigInt): UInt = {
    val w = (x - y + 1).toInt
    if (isLit()) UInt((litValue >> y.toInt) & ((BigInt(1) << w) - 1), w)
    else {
      val d = UInt(width = w)
      pushCommand(DefPrim(d.defd.cid, d.toType, BitsExtractOp, Seq(this.ref), Seq(x, y)))
      d
    }
  }
  final def apply(x: Int, y: Int): UInt =
    apply(BigInt(x), BigInt(y))
  final def apply(x: UInt, y: UInt): UInt =
    apply(x.litValue(), y.litValue())

  def maxWidth(other: Bits, amt: Int): Int = 
    if (getWidth >= 0 && other.getWidth >= 0) ((getWidth max other.getWidth) + amt) else -1
  override def sumWidth(amt: BigInt): Int = if (getWidth >= 0) (getWidth + amt).toInt else -1
  def sumWidth(other: Bits, amt: BigInt): Int =
    if (getWidth >= 0 && other.getWidth >= 0) (getWidth + other.getWidth + amt).toInt else -1
  def sumPow2Width(other: Bits): Int =
    if (getWidth >= 0 && other.getWidth >= 0) (getWidth + (1 << other.getWidth)).toInt else -1

  def :=(other: Bits) = 
    pushCommand(Connect(this.lref, other.ref))

  protected[Chisel] def unop(op: PrimOp, width: Int): this.type = {
    val d = cloneTypeWidth(width)
    pushCommand(DefPrim(d.defd.cid, d.toType, op, Seq(this.ref), NoLits))
    d
  }
  protected[Chisel] def binop(op: PrimOp, other: BigInt, width: Int): this.type = {
    val d = cloneTypeWidth(width)
    pushCommand(DefPrim(d.defd.cid, d.toType, op, Seq(this.ref), Seq(other)))
    d
  }
  protected[Chisel] def binop(op: PrimOp, other: Bits, width: Int): this.type = {
    val d = cloneTypeWidth(width)
    pushCommand(DefPrim(d.defd.cid, d.toType, op, Seq(this.ref, other.ref), NoLits))
    d
  }
  protected[Chisel] def compop(op: PrimOp, other: Bits): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d.defd.cid, d.toType, op, Seq(this.ref, other.ref), NoLits))
    d
  }

  def unary_- : Bits = Bits(0) - this
  def unary_-% : Bits = Bits(0) -% this
  def +& (other: Bits): Bits = binop(AddOp, other, maxWidth(other, 1))
  def + (other: Bits): Bits = this +% other
  def +% (other: Bits): Bits = binop(AddModOp, other, maxWidth(other, 0))
  def -& (other: Bits): Bits = binop(SubOp, other, maxWidth(other, 1))
  def -% (other: Bits): Bits = binop(SubModOp, other, maxWidth(other, 0))
  def - (other: Bits): Bits = this -% other
  def * (other: Bits): Bits = binop(TimesOp, other, sumWidth(other, 0))
  def / (other: Bits): Bits = binop(DivideOp, other, sumWidth(0))
  def % (other: Bits): Bits = binop(ModOp, other, sumWidth(0))
  def << (other: BigInt): Bits = binop(ShiftLeftOp, other, sumWidth(other))
  def << (other: Int): Bits = this << BigInt(other)
  def << (other: Bits): Bits = binop(DynamicShiftLeftOp, other, sumPow2Width(other))
  def >> (other: BigInt): Bits = binop(ShiftRightOp, other, sumWidth(-other))
  def >> (other: Int): Bits = this >> BigInt(other)
  def >> (other: Bits): Bits = binop(DynamicShiftRightOp, other, sumWidth(0))
  def unary_~ : Bits = unop(BitNotOp, sumWidth(0))
  def pad (other: BigInt): Bits = binop(PadOp, other, other.toInt)

  def & (other: Bits): Bits = binop(BitAndOp, other, maxWidth(other, 0))
  def | (other: Bits): Bits = binop(BitOrOp, other, maxWidth(other, 0))
  def ^ (other: Bits): Bits = binop(BitXorOp, other, maxWidth(other, 0))
  def ## (other: Bits): Bits = Cat(this, other)

  def < (other: Bits): Bool = compop(LessOp, other)
  def > (other: Bits): Bool = compop(GreaterOp, other)
  def === (other: Bits): Bool = compop(EqualOp, other)
  def != (other: Bits): Bool = compop(NotEqualOp, other)
  def <= (other: Bits): Bool = compop(LessEqOp, other)
  def >= (other: Bits): Bool = compop(GreaterEqOp, other)
  def unary_! : Bool = this === Bits(0)

  private def bits_redop(op: PrimOp): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d.defd.cid, d.toType, op, Seq(this.ref), NoLits))
    d
  }

  def === (that: BitPat): Bool = that === this
  def != (that: BitPat): Bool = that != this

  def orR = !(this === Bits(0))
  def andR = (this === Bits(-1))
  def xorR = bits_redop(XorReduceOp)

  def bitSet(off: UInt, dat: Bits): Bits = {
    val bit = UInt(1, 1) << off
    this & ~bit | dat.toSInt & bit
  }

  def toBools: Vec[Bool] = Vec.tabulate(this.getWidth)(i => this(i))

  def asSInt(): SInt
  def asUInt(): UInt
  def toSInt(): SInt
  def toUInt(): UInt
  def toBool(): Bool = this(0)
}

abstract trait Num[T <: Data] {
  // def << (b: T): T;
  // def >> (b: T): T;
  //def unary_-(): T;
  def +  (b: T): T;
  def *  (b: T): T;
  def /  (b: T): T;
  def %  (b: T): T;
  def -  (b: T): T;
  def <  (b: T): Bool;
  def <= (b: T): Bool;
  def >  (b: T): Bool;
  def >= (b: T): Bool;

  def min(b: T): T = Mux(this < b, this.asInstanceOf[T], b)
  def max(b: T): T = Mux(this < b, b, this.asInstanceOf[T])
}

class UInt(dir: Direction, width: Int, lit: Option[LitArg] = None) extends Bits(dir, width, lit) with Num[UInt] {
  override def cloneTypeWidth(w: Int): this.type =
    new UInt(dir, w).asInstanceOf[this.type]

  def toType: Kind = 
    UIntType(if (width == -1) UnknownWidth() else IntWidth(width), isFlipVar)

  override def makeLit(value: BigInt, width: Int): this.type = 
    UInt(value, width).asInstanceOf[this.type]

  override def unary_- = UInt(0) - this
  override def unary_-% = UInt(0) -% this
  def +& (other: UInt): UInt = binop(AddOp, other, maxWidth(other, 1))
  def + (other: UInt): UInt = this +% other
  def +% (other: UInt): UInt = binop(AddModOp, other, maxWidth(other, 0))
  def -& (other: UInt): UInt = binop(SubOp, other, maxWidth(other, 1))
  def - (other: UInt): UInt = this -% other
  def -% (other: UInt): UInt = binop(SubModOp, other, maxWidth(other, 0))
  def * (other: UInt): UInt = binop(TimesOp, other, sumWidth(other, 0))
  def / (other: UInt): UInt = binop(DivideOp, other, sumWidth(0))
  def % (other: UInt): UInt = binop(ModOp, other, sumWidth(0))
  override def << (other: BigInt): UInt = binop(ShiftLeftOp, other, sumWidth(other))
  override def << (other: Int): UInt = this << BigInt(other)
  def << (other: UInt): UInt = binop(DynamicShiftLeftOp, other, sumPow2Width(other))
  override def >> (other: BigInt): UInt = binop(ShiftRightOp, other, sumWidth(-other))
  override def >> (other: Int): UInt = this >> BigInt(other)
  def >> (other: UInt): UInt = binop(DynamicShiftRightOp, other, sumWidth(0))

  override def unary_~ : UInt = unop(BitNotOp, sumWidth(0))
  def & (other: UInt): UInt = binop(BitAndOp, other, maxWidth(other, 0))
  def | (other: UInt): UInt = binop(BitOrOp, other, maxWidth(other, 0))
  def ^ (other: UInt): UInt = binop(BitXorOp, other, maxWidth(other, 0))
  def ## (other: UInt): UInt = Cat(this, other)

  def < (other: UInt): Bool = compop(LessOp, other)
  def > (other: UInt): Bool = compop(GreaterOp, other)
  def === (other: UInt): Bool = compop(EqualOp, other)
  def != (other: UInt): Bool = compop(NotEqualOp, other)
  def <= (other: UInt): Bool = compop(LessEqOp, other)
  def >= (other: UInt): Bool = compop(GreaterEqOp, other)

  override def pad (other: BigInt): UInt = binop(PadOp, other, other.toInt)

  def zext(): SInt = {
    val x = SInt(width = getWidth + 1)
    pushCommand(DefPrim(x.defd.cid, x.toType, ConvertOp, Seq(ref), NoLits))
    x
  }

  def asSInt(): SInt = {
    val x = SInt(width = getWidth)
    pushCommand(DefPrim(x.defd.cid, x.toType, AsSIntOp, Seq(ref), NoLits))
    x
  }

  def toSInt(): SInt = asSInt()
  def toUInt(): UInt = this
  def asUInt(): UInt = this
}

trait UIntFactory {
  def apply(dir: Direction = OUTPUT, width: Int = -1) = 
    new UInt(dir, width)
  def apply(value: BigInt, width: Int) = {
    val w = if (width == -1) (1 max value.bitLength) else width
    new UInt(NO_DIR, w, Some(ULit(value, w)))
  }
  def apply(value: BigInt): UInt = apply(value, -1)
  def apply(n: String, width: Int): UInt = {
    val bitsPerDigit = if (n(0) == 'b') 1 else if (n(0) == 'h') 4 else -1
    apply(Literal.stringToVal(n(0), n.substring(1, n.length)),
          if (width == -1) (bitsPerDigit * (n.length-1)) else width)
  }
  def apply(n: String): UInt = apply(n, -1)
}

// Bits constructors are identical to UInt constructors.
object Bits extends UIntFactory
object UInt extends UIntFactory

class SInt(dir: Direction, width: Int, lit: Option[LitArg] = None) extends Bits(dir, width, lit) with Num[SInt] {
  override def cloneTypeWidth(w: Int): this.type =
    new SInt(dir, w).asInstanceOf[this.type]
  def toType: Kind = 
    SIntType(if (width == -1) UnknownWidth() else IntWidth(width), isFlipVar)

  override def makeLit(value: BigInt, width: Int): this.type =
    SInt(value, width).asInstanceOf[this.type]

  override def unary_- : SInt = SInt(0, getWidth) - this
  override def unary_-% : SInt = SInt(0, getWidth) -% this
  def +& (other: SInt): SInt = binop(AddOp, other, maxWidth(other, 1))
  def +% (other: SInt): SInt = binop(AddModOp, other, maxWidth(other, 0))
  def + (other: SInt): SInt = this +% other
  def -& (other: SInt): SInt = binop(SubOp, other, maxWidth(other, 1))
  def -% (other: SInt): SInt = binop(SubModOp, other, maxWidth(other, 0))
  def - (other: SInt): SInt = this -% other
  def * (other: SInt): SInt = binop(TimesOp, other, sumWidth(other, 0))
  def / (other: SInt): SInt = binop(DivideOp, other, sumWidth(0))
  def % (other: SInt): SInt = binop(ModOp, other, sumWidth(0))
  override def << (other: BigInt): SInt = binop(ShiftLeftOp, other, sumWidth(other))
  override def << (other: Int): SInt = this << BigInt(other)
  def << (other: UInt): SInt = binop(DynamicShiftLeftOp, other, sumPow2Width(other))
  override def >> (other: BigInt): SInt = binop(ShiftRightOp, other, sumWidth(-other))
  override def >> (other: Int): SInt = this >> BigInt(other)
  def >> (other: UInt): SInt = binop(DynamicShiftRightOp, other, sumWidth(0))

  override def unary_~ : SInt = unop(BitNotOp, sumWidth(0))
  def & (other: SInt): SInt = binop(BitAndOp, other, maxWidth(other, 0))
  def | (other: SInt): SInt = binop(BitOrOp, other, maxWidth(other, 0))
  def ^ (other: SInt): SInt = binop(BitXorOp, other, maxWidth(other, 0))

  def < (other: SInt): Bool = compop(LessOp, other)
  def > (other: SInt): Bool = compop(GreaterOp, other)
  def === (other: SInt): Bool = compop(EqualOp, other)
  def != (other: SInt): Bool = compop(NotEqualOp, other)
  def <= (other: SInt): Bool = compop(LessEqOp, other)
  def >= (other: SInt): Bool = compop(GreaterEqOp, other)
  def abs(): UInt = Mux(this < SInt(0), (-this).toUInt, this.toUInt)

  override def pad (other: BigInt): SInt = binop(PadOp, other, other.toInt)

  def asUInt(): UInt = {
    val x = UInt(width = getWidth)
    pushCommand(DefPrim(x.defd.cid, x.toType, AsUIntOp, Seq(ref), NoLits))
    x
  }
  def toUInt(): UInt = asUInt()
  def asSInt(): SInt = this
  def toSInt(): SInt = this
}

object SInt {
  def apply(dir: Direction = OUTPUT, width: Int = -1) = 
    new SInt(dir, width)
  def apply(value: BigInt, width: Int) = {
    val w = if (width == -1) 1 + value.bitLength else width
    new SInt(NO_DIR, w, Some(SLit(value, w)))
  }
  def apply(value: BigInt): SInt = apply(value, -1)
  def apply(n: String, width: Int): SInt =
    apply(Literal.stringToVal(n(0), n.substring(1, n.length)), width)
  def apply(n: String): SInt = apply(n, -1)
}

class Bool(dir: Direction, lit: Option[LitArg] = None) extends UInt(dir, 1, lit) {
  override def cloneTypeWidth(w: Int): this.type = new Bool(dir).asInstanceOf[this.type]

  override def makeLit(value: BigInt, width: Int): this.type =
    Bool(value).asInstanceOf[this.type]

  def & (other: Bool): Bool = super.&(other).asInstanceOf[Bool]
  def | (other: Bool): Bool = super.|(other).asInstanceOf[Bool]
  def ^ (other: Bool): Bool = super.^(other).asInstanceOf[Bool]
  override def unary_~ : Bool = super.unary_~.asInstanceOf[Bool]

  def || (that: Bool): Bool = this | that
  def && (that: Bool): Bool = this & that
}
object Bool {
  def apply(dir: Direction) : Bool = 
    new Bool(dir)
  def apply() : Bool = 
    apply(NO_DIR)
  def apply(value: BigInt) =
    new Bool(NO_DIR, Some(ULit(value, 1)))
  def apply(value: Boolean) : Bool = apply(if (value) 1 else 0)
}

object Mux {
  def apply[T <: Data](cond: Bool, con: T, alt: T): T = {
    val w = Wire(alt, init = alt)
    when (cond) {
      w := con
    }
    w
  }
}

object Cat {
  def apply[T <: Bits](a: T, r: T*): UInt = apply(a :: r.toList)
  def apply[T <: Bits](r: Seq[T]): UInt = {
    if (r.tail.isEmpty) r.head.asUInt
    else {
      val left = apply(r.slice(0, r.length/2))
      val right = apply(r.slice(r.length/2, r.length))
      val w = left.sumWidth(right, 0)
      if (left.isLit && right.isLit) {
        UInt((left.litValue() << right.getWidth) | right.litValue(), w)
      } else {
        val d = UInt(width = w)
        pushCommand(DefPrim(d.cid, d.toType, ConcatOp, Seq(left.ref, right.ref), NoLits))
        d
      }
    }
  }
}

object Bundle {
  val keywords = HashSet[String]("elements", "flip", "toString",
    "flatten", "binding", "asInput", "asOutput", "unary_$tilde",
    "unary_$bang", "unary_$minus", "cloneType", "clone",
    "toUInt", "toBits",
    "toBool", "toSInt", "asDirectionless")
  def apply[T <: Bundle](b: => T)(implicit p: Parameters): T = {
    Driver.parStack.push(p.push)
    val res = b
    Driver.parStack.pop
    res
  }
  def apply[T <: Bundle](b: => T,  f: PartialFunction[Any,Any]): T = {
    val q = params.alterPartial(f)
    apply(b)(q)
  }
  private def params = if(Driver.parStack.isEmpty) Parameters.empty else Driver.parStack.top
}

class Bundle(dirArg: Direction = NO_DIR) extends Aggregate(dirArg) { 
  def toPorts: Seq[Port] =
    elements.map(_._2.toPort).toSeq
  def toType: BundleType = 
    BundleType(this.toPorts, isFlipVar)

  override def flatten: IndexedSeq[Bits] =
    sortedElts.map(_._2.flatten).reduce(_ ++ _)

  lazy val elements: ListMap[String, Data] = ListMap(sortedElts:_*)

  private lazy val sortedElts = {
    val elts = ArrayBuffer[(String, Data)]()
    for (m <- getClass.getMethods) {
      val name = m.getName
      val rtype = m.getReturnType
      val isInterface = classOf[Data].isAssignableFrom(rtype)
      if (m.getParameterTypes.isEmpty &&
          !isStatic(m.getModifiers) &&
          isInterface &&
          !(Bundle.keywords contains name)) {
        m.invoke(this) match {
          case data: Data => elts += name -> data
          case _ =>
        }
      }
    }
    elts sortWith (_._2._id < _._2._id)
  }
  override def collectElts =
    sortedElts.foreach(e => setFieldForId(cid, e._2.cid, e._1))

  override def cloneType : this.type = {
    try {
      val constructor = this.getClass.getConstructors.head
      val res = constructor.newInstance(Seq.fill(constructor.getParameterTypes.size)(null):_*)
      res.asInstanceOf[this.type]
    } catch {
      case npe: java.lang.reflect.InvocationTargetException if npe.getCause.isInstanceOf[java.lang.NullPointerException] =>
        ChiselError.error(s"Parameterized Bundle ${this.getClass} needs cloneType method. You are probably using an anonymous Bundle object that captures external state and hence is un-cloneTypeable")
        this
      case e: java.lang.Exception =>
        ChiselError.error(s"Parameterized Bundle ${this.getClass} needs cloneType  method")
        this
    }
  }
}

object Module {
  def apply[T <: Module](bc: => T)(implicit p: Parameters = params): T = {
    Driver.modStackPushed = true
    Driver.parStack.push(p.push)
    val m = bc
    val cmd = popCommands
    popScope
    popModule
    m.setRefs
    val ports = m.io.toPorts
    val component = UniqueComponent(m.name, ports, cmd)
    components += component
    pushCommand(DefInstance(m.defd.cid, component.name))
    Driver.parStack.pop
    m
  }
  def apply[T <: Module](m: => T, f: PartialFunction[Any,Any]): T = {
    val q = params.alterPartial(f)
    apply(m)(q)
  }
  private def params = if(Driver.parStack.isEmpty) Parameters.empty else Driver.parStack.top
}

abstract class Module(private[Chisel] _reset: Bool = null) extends Id {
  private[Chisel] val _parent = modulez.headOption
  private[Chisel] val _nodes = ArrayBuffer[Data]()

  pushModule(this)
  pushScope
  pushCommands
  addModule(this)

  val params = Module.params
  params.path = this.getClass :: params.path

  def io: Bundle
  def ref = getRefForId(cid)
  def lref = ref
  val reset = if (_reset == null) Bool().defd else _reset
  setRefForId(reset.cid, "reset")

  def name = {
    // getClass.getName.replace('.', '_')
    getClass.getName.split('.').last
  }
  def debugName: String = (_parent match {
      case Some(p) => p.debugName + "."
      case None => ""
    }) + getRefForId(cid).debugName

  private def setRefs: Unit = {
    val valNames = HashSet[String](getClass.getDeclaredFields.map(_.getName):_*)
    def isPublicVal(m: java.lang.reflect.Method) =
      m.getParameterTypes.isEmpty && valNames.contains(m.getName) && isPublic(m.getModifiers)

    _nodes.foreach(_.collectElts)
    _nodes.clear

    setRefForId(io.cid, "this")

    for (m <- getClass.getDeclaredMethods; if isPublicVal(m)) {
      m.invoke(this) match {
        case module: Module =>
          setRefForId(module.cid, m.getName)
          module.setRefs
        case bundle: Bundle =>
          if (m.getName != "io") {
            setRefForId(bundle.cid, m.getName)
          }
        case mem: Mem[_] =>
          setRefForId(mem.t.cid, m.getName)
        case vec: Vec[_] =>
          setRefForId(vec.cid, m.getName)
        case data: Data =>
          setRefForId(data.cid, m.getName)
        // ignore anything not of those types
        case _ => null
      }
    }
  }

  // TODO: actually implement these
  def assert(cond: Bool, msg: String): Unit = {}
  def printf(message: String, args: Bits*): Unit = {}
}

// TODO: actually implement BlackBox (this hack just allows them to compile)
abstract class BlackBox(private[Chisel] _reset: Bool = null) extends Module(_reset) {
  def setVerilogParameters(s: String): Unit = {}
}

object when {
  private[Chisel] def execBlock(block: => Unit): Command = {
    pushScope
    pushCommands
    block
    val cmd = popCommands
    popScope
    cmd
  }
  def apply(cond: => Bool)(block: => Unit): when = {
    new when(cond)( block )
  }
}

class when(cond: => Bool)(block: => Unit) {
  def elsewhen (cond: => Bool)(block: => Unit): when = {
    pushCommands
    val res = new when(cond) ( block )
    this.cmd.alt = popCommands
    res
  }

  def otherwise (block: => Unit) {
   this.cmd.alt = when.execBlock(block)
  }

  // Capture any commands we need to set up the conditional test.
  pushCommands
  val pred = cond.ref
  val prep = popCommands
  val conseq  = when.execBlock(block)
  // Assume we have an empty alternate clause.
  //  elsewhen and otherwise will update it if that isn't the case.
  val cmd = Conditionally(prep, pred, conseq, EmptyCommand())
  pushCommand(cmd)
}


/// CHISEL IR EMITTER

class Emitter {
  private var indenting = 0
  def withIndent(f: => String) = {
    indenting += 1
    val res = f
    indenting -= 1
    res
  }

  def newline = "\n" + ("  " * indenting)
  def join(parts: Seq[String], sep: String): StringBuilder =
    parts.tail.foldLeft(new StringBuilder(parts.head))((s, p) => s ++= sep ++= p)
  def join0(parts: Seq[String], sep: String): StringBuilder =
    parts.foldLeft(new StringBuilder)((s, p) => s ++= sep ++= p)
  def emitDir(e: Direction, isTop: Boolean): String =
    if (isTop) (e.name + " ") else if (e == INPUT) "flip " else ""
  def emit(e: PrimOp): String = e.name
  def emit(e: Arg): String = e.fullname
  def emitPort(e: Port, isTop: Boolean): String =
    emitDir(e.dir, isTop) + getRefForId(e.id).name + " : " + emitType(e.kind)
  def emit(e: Width): String = {
    e match {
      case e: UnknownWidth => ""
      case e: IntWidth => "<" + e.value.toString + ">"
    }
  }
  def emitType(e: Kind): String = {
    e match {
      case e: UnknownType => "?"
      case e: UIntType => "UInt" + emit(e.width)
      case e: SIntType => "SInt" + emit(e.width)
      case e: BundleType => "{" + join(e.ports.map(x => emitPort(x, false)), ", ") + "}"
      case e: VectorType => emitType(e.kind) + "[" + e.size + "]"
    }
  }
  def emit(e: Command): String = {
    def maybeWidth (w: Int) = if (w == -1) "<?>" else ("<" + w + ">")
    e match {
      case e: DefUInt => "node " + e.name + " = UInt" + maybeWidth(e.width) + "(" + e.value + ")"
      case e: DefSInt => "node " + e.name + " = SInt" + maybeWidth(e.width) + "(" + e.value + ")"
      case e: DefFlo => "node " + e.name + " = Flo(" + e.value + ")"
      case e: DefDbl => "node " + e.name + " = Dbl(" + e.value + ")"
      case e: DefPrim =>
        "node " + e.name + " = " + emit(e.op) + "(" + join(e.args.map(x => emit(x)) ++ e.lits.map(x => x.toString), ", ") + ")"
      case e: DefWire => "wire " + e.name + " : " + emitType(e.kind)
      case e: DefRegister => "reg " + e.name + " : " + emitType(e.kind)
      case e: DefMemory => "cmem " + e.name + " : " + emitType(e.kind) + "[" + e.size + "]";
      case e: DefSeqMemory => "smem " + e.name + " : " + emitType(e.kind) + "[" + e.size + "]";
      case e: DefAccessor => "accessor " + e.name + " = " + emit(e.source) + "[" + emit(e.index) + "]"
      case e: DefInstance => {
        val mod = modules(e.id)
        // update all references to the modules ports
        setRefForId(mod.io.cid, e.name, true)
        "inst " + e.name + " of " + e.module
      }
      case e: Conditionally => {
        val prefix = if (!e.prep.isInstanceOf[EmptyCommand]) {
          newline + emit(e.prep) + newline
        } else {
          ""
        }
        val suffix = if (!e.alt.isInstanceOf[EmptyCommand]) {
          newline + "else : " + withIndent{ newline + emit(e.alt) }
        } else {
          ""
        }
        prefix + "when " + emit(e.pred) + " : " + withIndent{ emit(e.conseq) } + suffix
      }
      case e: Begin => join0(e.body.map(x => emit(x)), newline).toString
      case e: Connect => emit(e.loc) + " := " + emit(e.exp)
      case e: BulkConnect => emit(e.loc1) + " <> " + emit(e.loc2)
      case e: ConnectInit => "on-reset " + emit(e.loc) + " := " + emit(e.exp)
      case e: ConnectInitIndex => "on-reset " + emit(e.loc) + "[" + e.index + "] := " + emit(e.exp)
      case e: EmptyCommand => "skip"
    }
  }
  def emit(e: Component): String =  {
    withIndent{ "module " + e.name + " : " +
      join0(e.ports.map(x => emitPort(x, true)), newline) +
      newline + emit(e.body) }
  }
  def emit(e: Circuit): String = 
    withIndent{ "circuit " + e.main + " : " + join0(e.components.map(x => emit(x)), newline) }
}
