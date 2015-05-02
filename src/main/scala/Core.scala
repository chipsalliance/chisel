package Chisel
import scala.collection.mutable.{ArrayBuffer, Stack, HashSet, HashMap}
import java.lang.reflect.Modifier._
import scala.math.max

class GenSym {
  var counter = -1
  def next(name: String): String = {
    counter += 1
    name + "_" + counter
  }
}

object Builder {
  val components = new ArrayBuffer[Component]()
  val genSym = new GenSym()
  val scopes = new Stack[HashSet[String]]()
  def scope = scopes.top
  val switchKeyz = new Stack[Stack[Bits]]()
  def switchKeys = switchKeyz.top
  def pushScope = {
    scopes.push(new HashSet[String]())
    switchKeyz.push(new Stack[Bits]())
  }
  def popScope = {
    scopes.pop()
    switchKeyz.pop()
  }
  val modules = new HashMap[String,Module]()
  def addModule(mod: Module) {
    modules(mod.id) = mod
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
  def UniqueComponent(name: String, ports: Array[Port], body: Command) = {
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
      Begin(cmds.toArray)
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

  def setRefForId(id: String, name: String, overwrite: Boolean = false) {
    if (overwrite || !refmap.contains(id)) {
      refmap(id) = Ref(name)
    }
  }

  def setFieldForId(parentid: String, id: String, name: String) {
    refmap(id) = Slot(Alias(parentid), name)
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
    (Circuit(components.toArray, components.last.name), mod)
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
  val AndOp = PrimOp("bit-and")
  val OrOp = PrimOp("bit-or")
  val BitAndOp = PrimOp("bit-and")
  val BitOrOp = PrimOp("bit-or")
  val BitXorOp = PrimOp("bit-xor")
  val BitNotOp = PrimOp("bit-not")
  val NotOp = PrimOp("bit-not")
  val ConcatOp = PrimOp("cat")
  val BitSelectOp = PrimOp("bit")
  val BitsExtractOp = PrimOp("bits")
  val LessOp = PrimOp("lt")
  val LessEqOp = PrimOp("leq")
  val GreaterOp = PrimOp("gt")
  val GreaterEqOp = PrimOp("geq")
  val EqualOp = PrimOp("eq")
  val PatternEqualOp = PrimOp("pattern-equal")
  val PadOp = PrimOp("Pad")
  val NotEqualOp = PrimOp("neq")
  val NegOp = PrimOp("neg")
  val MultiplexOp = PrimOp("mux")
  val AndReduceOp = PrimOp("bit-and-reduce")
  val OrReduceOp = PrimOp("bit-or-reduce")
  val XorReduceOp = PrimOp("bit-xor-reduce")
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
    if (imm_debugName == "this") name else imm_debugName + "_" + name
  }
}
case class Index(val imm: Immediate, val value: Int) extends Immediate {
  def name = "[" + value + "]"
  def fullname = imm.fullname + "[" + value + "]"
  override def debugName = imm.debugName + "_" + value
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
case class BundleType(val ports: Array[Port], flip: Boolean) extends Kind(flip);
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
case class DefMInt(val id: String, val value: String, val width: Int) extends Definition;
case class DefPrim(val id: String, val kind: Kind, val op: PrimOp, val args: Array[Arg], val lits: Array[BigInt]) extends Definition;
case class DefPrimPad(val id: String, val kind: Kind, val op: PrimOp, val args: Array[Arg], val lits: Array[BigInt]) extends Definition;
case class DefWire(val id: String, val kind: Kind) extends Definition;
case class DefRegister(val id: String, val kind: Kind) extends Definition;
case class DefMemory(val id: String, val kind: Kind, val size: Int) extends Definition;
case class DefAccessor(val id: String, val source: Alias, val direction: Direction, val index: Arg) extends Definition;
case class DefInstance(val id: String, val module: String) extends Definition;
case class Conditionally(val pred: Arg, val conseq: Command, val alt: Command) extends Command;
case class Begin(val body: Array[Command]) extends Command();
case class Connect(val loc: Alias, val exp: Arg) extends Command;
case class ConnectPad(val loc: Alias, val exp: Arg) extends Command;
case class ConnectInit(val loc: Alias, val exp: Arg) extends Command;
case class ConnectInitIndex(val loc: Alias, val index: Int, val exp: Arg) extends Command;
case class EmptyCommand() extends Command;

case class Component(val name: String, val ports: Array[Port], val body: Command);
case class Circuit(val components: Array[Component], val main: String);

object Commands {
  val NoLits = Array[BigInt]()
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
  val id = genSym.next("id")
  var isDef_ = false
  def defd: this.type = {
    isDef_ = true
    this
  }
  def isDef = isDef_
}

abstract class Data(dirArg: Direction) extends Id {
  val mod = getComponent()
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
    pushCommand(Connect(this.lref, other.ref))
  def cloneType: this.type
  def cloneTypeWidth(width: Int): this.type
  def lref: Alias = 
    Alias(id)
  def ref: Arg = 
    if (isLitValue) litArg() else Alias(id)
  def name = getRefForId(id).name
  def debugName = mod.debugName + "." + getRefForId(id).debugName
  def litArg(): LitArg = null
  def litValue(): BigInt = -1
  def isLitValue(): Boolean = false
  def setLitValue(x: LitArg) {  }
  def getWidth: Int
  def flatten: Array[Bits]
  def fromBits(n: Bits): this.type = {
    val res = this.cloneType
    var i = 0
    val wire = Wire(res)
    for (x <- wire.flatten.reverse) {
      x := n(i + x.getWidth-1, i)
      i += x.getWidth
    }
    res
  }
  def toBits: Bits = {
    val elts = this.flatten
    Cat(elts.head, elts.tail:_*)
  }
  def makeLit(value: BigInt, width: Int): this.type = {
    val x = cloneType
    x.fromBits(Bits(value, width))
    x
  }
  def toBool(): Bool = 
    chiselCast(this){Bool()}

  def toPort: Port = Port(id, dir, toType)
  def collectElts: Unit
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
    x.collectElts
    pushCommand(DefWire(x.defd.id, x.toType))
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
    pushCommand(DefRegister(x.defd.id, x.toType))
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
    pushCommand(DefMemory(mt.defd.id, mt.toType, size))
    mem
  }
}

class Mem[T <: Data](val t: T, val n: Int) /* with VecLike[T]  */ { // TODO: VECLIKE
  def apply(idx: Bits): T = {
    val x = t.cloneType
    pushCommand(DefAccessor(x.defd.id, Alias(t.id), NO_DIR, idx.ref))
    x
  }
  def name = getRefForId(t.id).name
  def debugName = t.mod.debugName + "." + getRefForId(t.id).debugName
}

object Vec {
  def apply[T <: Data](gen: T, n: Int): Vec[T] = 
    new Vec((0 until n).map(i => gen.cloneType))
  def apply[T <: Data](elts: Iterable[T]): Vec[T] = {
    val vec = new Vec[T](elts.map(e => elts.head.cloneType))
    vec.collectElts
    val isDef = true || elts.head.isDef
    if (vec.isReg)
      throw new Exception("Vec of Reg Deprecated.")
    if (isDef) {
      pushCommand(DefWire(vec.defd.id, vec.toType))
      var i = 0
      for (elt <- elts) {
        vec(i) := elt
        i += 1
      }
    }
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

class Vec[T <: Data](val elts: Iterable[T], dirArg: Direction = NO_DIR) extends Aggregate(dirArg) with VecLike[T] {
  val elt0 = elts.head
  val self = new ArrayBuffer[T]()
  self ++= elts
  override def isReg = elt0.isReg
  override def isFlip = {
    val isSubFlip = elt0.isFlip
    if (isFlipVar) !isSubFlip else isSubFlip
  }

  def apply(idx: UInt): T = {
    val x = elt0.cloneType
    pushCommand(DefAccessor(x.defd.id, Alias(id), NO_DIR, idx.ref))
    x
  }
  def apply(idx: Int): T = 
    self(idx)
  def toPorts: Array[Port] = 
    self.map(d => d.toPort).toArray
  def toType: Kind = 
    VectorType(self.size, elt0.toType, isFlipVar)
  override def cloneType: this.type = {
    val v = Vec(elt0.cloneType, self.size).asInstanceOf[this.type]
    v.collectElts
    v
  }
  def inits (f: (Int, T, (Int, T, T) => Unit) => Unit) = {
    var i = 0
    def doInit (index: Int, elt: T, init: T) =
      pushCommand(ConnectInitIndex(elt.lref, index, init.ref))
    for (d <- self) {
      f(i, d, doInit)
      i += 1;
    }
  }
  override def flatten: Array[Bits] = 
    self.map(_.flatten).reduce(_ ++ _)
  override def getWidth: Int = 
    flatten.map(_.getWidth).reduce(_ + _)

  def collectElts: Unit = {
    for (i <- 0 until self.size) {
      val elt = self(i)
      setIndexForId(id, elt.id, i)
      elt.collectElts
    }
  }

  def length: Int = self.size
}

trait VecLike[T <: Data] extends collection.IndexedSeq[T] {
  // def read(idx: UInt): T
  // def write(idx: UInt, data: T): Unit
  def apply(idx: UInt): T

  def forall(p: T => Bool): Bool = (this map p).fold(Bool(true))(_&&_)
  def exists(p: T => Bool): Bool = (this map p).fold(Bool(false))(_||_)
  // def contains[T <: Bits](x: T): Bool = this.exists(_ === x)
  def count(p: T => Bool): UInt = PopCount((this map p).toSeq)

  private def indexWhereHelper(p: T => Bool) = this map p zip (0 until length).map(i => UInt(i))
  def indexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p))
  def lastIndexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p).reverse)
  def onlyIndexWhere(p: T => Bool): UInt = Mux1H(indexWhereHelper(p))
}

object chiselCast {
  def apply[S <: Data, T <: Bits](i: S)(gen: => T): T = {
    val x = gen
    pushCommand(DefWire(x.defd.id, x.toType))
    val b = i.toBits
    pushCommand(Connect(x.lref, b.ref))
    x
  }
}

import Literal._

class MInt(val value: String, val width: Int) extends Data(NO_DIR) {
  def cloneTypeWidth(width: Int): this.type = cloneType
  def collectElts: Unit = { }
  override def dir: Direction = NO_DIR
  override def setDir(dir: Direction): Unit = { }
  override def toType: Kind = UIntType(UnknownWidth(), isFlip)
  override def getWidth: Int = width
  override def flatten: Array[Bits] = Array[Bits](Bits(0))
  override def cloneType: this.type = 
    new MInt(value, width).asInstanceOf[this.type]
  def fromInt(x: BigInt): MInt = MInt(x.toString(2), -1).asInstanceOf[this.type]
  val (bits, mask, swidth) = parseLit(value)
  def zEquals(other: Bits): Bool = 
    (Bits(toLitVal(mask, 2)) & other) === Bits(toLitVal(bits, 2))
  def === (other: Bits): Bool = zEquals(other)
  def != (other: Bits): Bool  = !zEquals(other)
}

object MInt {
  def mintLit(n: String, width: Int) = {
    assert(n(0) == 'b', "BINARY MINTS ONLY")
    new MInt(n.substring(1, n.length), width)
  }
  def apply(value: String, width: Int): MInt = mintLit(value, width)
  def apply(value: String): MInt = apply(value, -1)
}

abstract class Element(dirArg: Direction, val width: Int) extends Data(dirArg) {
  def collectElts: Unit = { }
  override def getWidth: Int = width
}

abstract class Bits(dirArg: Direction, width: Int) extends Element(dirArg, width) {
  private var litValueVar: Option[LitArg] = None

  override def litArg(): LitArg = litValueVar.get
  override def isLitValue(): Boolean = litValueVar.isDefined
  override def litValue(): BigInt = if (isLitValue) litValueVar.get.num else -1
  override def setLitValue(x: LitArg) { litValueVar = Some(x) }

  override def flatten: Array[Bits] = Array[Bits](this)

  final def apply(x: UInt): Bool = 
    apply(x.litValue())

  final def apply(x: BigInt): Bool = {
    val d = new Bool(dir)
    if (isLitValue())
      d.setLitValue(ULit((litValue() >> x.toInt) & 1, 1))
    else
      pushCommand(DefPrim(d.defd.id, d.toType, BitSelectOp, Array(this.ref), Array(x)))
    d
  }
  final def apply(x: UInt, y: UInt): UInt = 
    apply(x.litValue(), y.litValue())

  final def apply(x: BigInt, y: BigInt): UInt = {
    val w = (x - y + 1).toInt
    val d = UInt(width = w)
    if (isLitValue()) {
      val mask = (BigInt(1)<<d.getWidth)-BigInt(1)
      d.setLitValue(ULit((litValue() >> y.toInt) & mask, w))
    } else
      pushCommand(DefPrim(d.defd.id, d.toType, BitsExtractOp, Array(this.ref), Array(x, y)))
    d
  }

  def :=(other: Bits) = 
    pushCommand(ConnectPad(this.lref, other.ref))

  override def fromBits(n: Bits): this.type = {
    val res = Wire(this.cloneType)
    res := n
    res.asInstanceOf[this.type]
  }

  private def bits_unop(op: PrimOp): Bits = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref), NoLits))
    d
  }
  private def bits_binop(op: PrimOp, other: BigInt): Bits = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref), Array(other)))
    d.asInstanceOf[Bits]
  }
  private def bits_binop_pad(op: PrimOp, other: Bits): Bits = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrimPad(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def bits_binop(op: PrimOp, other: Bits): Bits = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def bits_compop_pad(op: PrimOp, other: Bits): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrimPad(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }

  // def unary_- = bits_unop(NegOp)
  // def unary_-% = bits_unop(NegModOp)
  def unary_- = Bits(0) - this
  def unary_-% = Bits(0) -% this
  def +& (other: Bits) = bits_binop_pad(AddOp, other)
  def + (other: Bits) = this +% other
  def +% (other: Bits) = bits_binop_pad(AddModOp, other)
  def -& (other: Bits) = bits_binop_pad(SubOp, other)
  def -% (other: Bits) = bits_binop_pad(SubModOp, other)
  def - (other: Bits) = this -% other
  def * (other: Bits) = bits_binop_pad(TimesOp, other)
  def / (other: Bits) = bits_binop_pad(DivideOp, other)
  def % (other: Bits) = bits_binop_pad(ModOp, other)
  def << (other: BigInt) = bits_binop(ShiftLeftOp, other)
  def << (other: Bits) = bits_binop(DynamicShiftLeftOp, other)
  def >> (other: BigInt) = bits_binop(ShiftRightOp, other)
  def >> (other: Bits) = bits_binop(DynamicShiftRightOp, other)
  def unary_~(): Bits = bits_unop(BitNotOp)

  def & (other: Bits) = {
    if (isLitValue() && other.isLitValue()) {
      val r = cloneTypeWidth(-1)
      r.setLitValue(ULit(litValue() & other.litValue()))
      r
    } else
      bits_binop_pad(BitAndOp, other)
  }
  def | (other: Bits) = bits_binop_pad(BitOrOp, other)
  def ^ (other: Bits) = bits_binop_pad(BitXorOp, other)

  def < (other: Bits) = bits_compop_pad(LessOp, other)
  def > (other: Bits) = bits_compop_pad(GreaterOp, other)
  def === (other: Bits) = {
    if (isLitValue() && other.isLitValue()) {
      val r = new Bool(dir)
      r.setLitValue(ULit(if (litValue() == other.litValue()) 1 else 0, 1))
      r
    } else
      bits_compop_pad(EqualOp, other)
  }
  def != (other: Bits) = bits_compop_pad(NotEqualOp, other)
  def <= (other: Bits) = bits_compop_pad(LessEqOp, other)
  def >= (other: Bits) = bits_compop_pad(GreaterEqOp, other)
  def pad (other: BigInt) = bits_binop(PadOp, other)

  private def bits_redop(op: PrimOp): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref), NoLits))
    d
  }

  def orR = bits_redop(OrReduceOp)
  def andR = bits_redop(AndReduceOp)
  def xorR = bits_redop(XorReduceOp)

  def bitSet(off: BigInt, dat: Bits): Bits = {
    val bit = UInt(1, 1) << off
    this & ~bit | dat.toSInt & bit
  }

  def toBools: Vec[Bool] = Vec.tabulate(this.getWidth)(i => this(i))

  def fromInt(x: BigInt): this.type;

  def toSInt(): SInt = chiselCast(this){SInt()};

  def toUInt(): UInt = chiselCast(this){UInt()};
}

import UInt._

object Bits {
  def apply(dir: Direction = OUTPUT, width: Int = -1) = new UInt(dir, width)
  def apply(value: BigInt, width: Int): UInt = uintLit(value, width)
  def apply(value: BigInt): UInt = apply(value, -1)
  def apply(n: String, width: Int): UInt =
    apply(stringToVal(n(0), n.substring(1, n.length)), width)
  def apply(n: String): UInt = apply(n, -1)
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

class UInt(dir: Direction, width: Int) extends Bits(dir, width) with Num[UInt] {
  override def cloneTypeWidth(w: Int): this.type =
    new UInt(dir, w).asInstanceOf[this.type]
  override def cloneType : this.type = cloneTypeWidth(width)

  def toType: Kind = 
    UIntType(if (width == -1) UnknownWidth() else IntWidth(width), isFlipVar)

  def fromInt(x: BigInt): this.type = UInt(x).asInstanceOf[this.type]

  override def makeLit(value: BigInt, width: Int): this.type = 
    UInt(value, width).asInstanceOf[this.type]

  private def uint_unop(op: PrimOp): UInt = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref), NoLits))
    d
  }
  private def uint_binop(op: PrimOp, other: UInt): UInt = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def uint_binop_pad(op: PrimOp, other: UInt): UInt = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrimPad(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def uint_binop(op: PrimOp, other: BigInt): UInt = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref), Array(other)))
    d
  }
  private def uint_compop_pad(op: PrimOp, other: UInt): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrimPad(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }

  // override def unary_- = uint_unop(NegOp)
  override def unary_- = UInt(0) - this
  override def unary_-% = UInt(0) -% this
  def +& (other: UInt) = uint_binop_pad(AddOp, other)
  def + (other: UInt) = this +% other
  def +% (other: UInt) = uint_binop_pad(AddModOp, other)
  def -& (other: UInt) = uint_binop_pad(SubOp, other)
  def - (other: UInt) = this -% other
  def -% (other: UInt) = uint_binop_pad(SubModOp, other)
  def * (other: UInt) = uint_binop_pad(TimesOp, other)
  def / (other: UInt) = uint_binop_pad(DivideOp, other)
  def % (other: UInt) = uint_binop_pad(ModOp, other)
  override def << (other: BigInt) = uint_binop(ShiftLeftOp, other)
  def << (other: UInt) = uint_binop(DynamicShiftLeftOp, other)
  override def >> (other: BigInt) = uint_binop(ShiftRightOp, other)
  def >> (other: UInt) = uint_binop(DynamicShiftRightOp, other)

  override def unary_~(): UInt = uint_unop(BitNotOp)
  def & (other: UInt) = {
    if (isLitValue() && other.isLitValue()) {
      val r = cloneTypeWidth(-1)
      r.setLitValue(ULit(litValue() & other.litValue()))
      r
    } else
      uint_binop_pad(BitAndOp, other)
  }
  def | (other: UInt) = uint_binop_pad(BitOrOp, other)
  def ^ (other: UInt) = uint_binop_pad(BitXorOp, other)

  def < (other: UInt) = uint_compop_pad(LessOp, other)
  def > (other: UInt) = uint_compop_pad(GreaterOp, other)
  def === (other: UInt) = {
    if (isLitValue() && other.isLitValue()) {
      val r = new Bool(dir)
      r.setLitValue(ULit(if (litValue() == other.litValue()) 1 else 0, 1))
      r
    } else
      uint_compop_pad(EqualOp, other)
  }
  def != (other: UInt) = uint_compop_pad(NotEqualOp, other)
  def <= (other: UInt) = uint_compop_pad(LessEqOp, other)
  def >= (other: UInt) = uint_compop_pad(GreaterEqOp, other)

  override def pad (other: BigInt) = uint_binop(PadOp, other)
}

object UInt {
  def apply(dir: Direction = OUTPUT, width: Int = -1) = 
    new UInt(dir, width)
  def uintLit(value: BigInt, width: Int) = {
    val w = if(width == -1) max(bitLength(value), 1) else width
    val b = new UInt(NO_DIR, w)
    b.setLitValue(ULit(value, w))
    // pushCommand(DefUInt(b.defd.id, value, w))
    b
  }
  def apply(value: BigInt, width: Int): UInt = uintLit(value, width)
  def apply(value: BigInt): UInt = apply(value, -1)
  def apply(n: String, width: Int): UInt =
    apply(stringToVal(n(0), n.substring(1, n.length)), width)
  def apply(n: String): UInt = apply(n, -1)
}

class SInt(dir: Direction, width: Int) extends Bits(dir, width) with Num[SInt] {
  override def cloneTypeWidth(w: Int): this.type =
    new SInt(dir, w).asInstanceOf[this.type]
  override def cloneType: this.type = cloneTypeWidth(width)
  def toType: Kind = 
    SIntType(if (width == -1) UnknownWidth() else IntWidth(width), isFlipVar)

  def fromInt(x: BigInt): this.type = SInt(x).asInstanceOf[this.type]

  override def makeLit(value: BigInt, width: Int): this.type =
    SInt(value, width).asInstanceOf[this.type]

  private def sint_unop(op: PrimOp): SInt = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref), NoLits))
    d
  }
  private def sint_binop(op: PrimOp, other: BigInt): Bits = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref), Array(other)))
    d
  }
  private def sint_binop(op: PrimOp, other: Bits): SInt = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrim(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def sint_binop_pad(op: PrimOp, other: SInt): SInt = {
    val d = cloneTypeWidth(-1)
    pushCommand(DefPrimPad(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }
  private def sint_compop_pad(op: PrimOp, other: SInt): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrimPad(d.defd.id, d.toType, op, Array(this.ref, other.ref), NoLits))
    d
  }

  // override def unary_- = sint_unop(NegOp)
  override def unary_- = SInt(0) - this
  override def unary_-% = SInt(0) -% this
  def +& (other: SInt) = sint_binop_pad(AddOp, other)
  def + (other: SInt) = sint_binop_pad(AddModOp, other)
  def -& (other: SInt) = sint_binop_pad(SubOp, other)
  def - (other: SInt) = sint_binop_pad(SubModOp, other)
  def * (other: SInt) = sint_binop_pad(TimesOp, other)
  def / (other: SInt) = sint_binop_pad(DivideOp, other)
  def % (other: SInt) = sint_binop_pad(ModOp, other)
  override def << (other: BigInt) = sint_binop(ShiftLeftOp, other)
  def << (other: UInt) = sint_binop(DynamicShiftLeftOp, other)
  override def >> (other: BigInt) = sint_binop(ShiftRightOp, other)
  def >> (other: UInt) = sint_binop(DynamicShiftRightOp, other)

  override def unary_~(): SInt = sint_unop(BitNotOp)
  def & (other: SInt) = sint_binop_pad(BitAndOp, other)
  def | (other: SInt) = sint_binop_pad(BitOrOp, other)
  def ^ (other: SInt) = sint_binop_pad(BitXorOp, other)

  def < (other: SInt) = sint_compop_pad(LessOp, other)
  def > (other: SInt) = sint_compop_pad(GreaterOp, other)
  def === (other: SInt) = {
    val r = sint_compop_pad(EqualOp, other)
    if (isLitValue() && other.isLitValue())
      r.setLitValue(SLit(if (litValue() == other.litValue()) 1 else 0))
    r
  }
  def != (other: SInt) = sint_compop_pad(NotEqualOp, other)
  def <= (other: SInt) = sint_compop_pad(LessEqOp, other)
  def >= (other: SInt) = sint_compop_pad(GreaterEqOp, other)

  override def pad (other: BigInt) = sint_binop(PadOp, other)
}

object SInt {
  def apply(dir: Direction = OUTPUT, width: Int = -1) = 
    new SInt(dir, width)
  def sintLit(value: BigInt, width: Int) = {
    val w = if (width == -1) bitLength(value) + 1 else width
    val b = new SInt(NO_DIR, w)
    b.setLitValue(SLit(value, w))
    // pushCommand(DefSInt(b.defd.id, value, w))
    b
  }
  def apply(value: BigInt, width: Int): SInt = sintLit(value, width)
  def apply(value: BigInt): SInt = apply(value, -1)
  def apply(n: String, width: Int): SInt =
    apply(stringToVal(n(0), n.substring(1, n.length)), width)
  def apply(n: String): SInt = apply(n, -1)
}

class Bool(dir: Direction) extends UInt(dir, 1) {
  override def cloneTypeWidth(w: Int): this.type = new Bool(dir).asInstanceOf[this.type]
  override def cloneType: this.type = cloneTypeWidth(1)

  override def fromInt(x: BigInt): this.type = Bool(x).asInstanceOf[this.type]

  override def makeLit(value: BigInt, width: Int): this.type =
    Bool(value).asInstanceOf[this.type]

  def || (other: Bool) = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d.defd.id, d.toType, OrOp, Array(this.ref, other.ref), NoLits))
    d
  }
  def && (other: Bool) = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d.defd.id, d.toType, AndOp, Array(this.ref, other.ref), NoLits))
    d
  }
  def unary_! (): Bool = {
    val d = new Bool(dir)
    pushCommand(DefPrim(d.defd.id, d.toType, NotOp, Array(this.ref), NoLits))
    d
  }
}
object Bool {
  def apply(dir: Direction) : Bool = 
    new Bool(dir)
  def apply() : Bool = 
    apply(NO_DIR)
  def apply(value: BigInt) : Bool = {
    val b = new Bool(NO_DIR)
    pushCommand(DefUInt(b.defd.id, value, 1))
    b
  }
  def apply(value: Boolean) : Bool = 
    apply(if (value) 1 else 0)
}

object Mux {
  def apply[T <: Data](cond: Bool, con: T, alt: T): T = {
    val r = con.cloneTypeWidth(-1)
    val args = Array(cond.ref, con.ref, alt.ref)
    con match {
      case t: Bits => 
        pushCommand(DefPrimPad(r.defd.id, r.toType, MultiplexOp, args, NoLits))
      case _ =>
        pushCommand(DefPrim(r.defd.id, r.toType, MultiplexOp, args, NoLits))
    }
    r.asInstanceOf[T]
  }
}

object Cat {
  def apply[T <: Bits](a: T, r: T*): T = apply(a :: r.toList)
  def apply[T <: Bits](r: Seq[T]): T = doCat(r)
  private def doCat[T <: Data](r: Seq[T]): T = {
    if (r.tail.isEmpty)
      r.head
    else {
      val l = doCat(r.slice(0, r.length/2))
      val h = doCat(r.slice(r.length/2, r.length))
      val isConst = (l.isLitValue() && h.isLitValue())
      val w = if (isConst) l.getWidth + h.getWidth else -1
      val d = l.cloneTypeWidth(w)
      if (isConst) {
        val c = (l.litValue() << h.getWidth) | h.litValue()
        // println("DO-CAT L = " + l.litValue() + " LW = " + l.getWidth + " H = " + h.litValue() + " -> " + c)

        d.setLitValue(ULit(c, w))
      } else
        pushCommand(DefPrim(d.id, d.toType, ConcatOp, Array(l.ref, h.ref), NoLits))
      d
    }
  }
}

object Bundle {
  val keywords = HashSet[String]("elements", "flip", "toString",
    "flatten", "binding", "asInput", "asOutput", "unary_$tilde",
    "unary_$bang", "unary_$minus", "cloneType", "toUInt", "toBits",
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
  def toPorts: Array[Port] = 
    elts.map(d => d.toPort).toArray
  def toType: BundleType = 
    BundleType(this.toPorts, isFlipVar)

  override def flatten: Array[Bits] = 
    elts.map(_.flatten).reduce(_ ++ _)
  override def getWidth: Int = 
    flatten.map(_.getWidth).reduce(_ + _)

  val elts = ArrayBuffer[Data]()
  def collectElts: Unit = {
    elts.clear()
    for (m <- getClass.getDeclaredMethods) {
      val name = m.getName

      val modifiers = m.getModifiers();
      val types = m.getParameterTypes()
      var isInterface = false;
      var isFound = false;
      val rtype = m.getReturnType();
      var c = rtype;
      val sc = Class.forName("Chisel.Data");
      do {
        if (c == sc) {
          isFound = true; isInterface = true;
        } else if (c == null || c == Class.forName("java.lang.Object")) {
          isFound = true; isInterface = false;
        } else {
          c = c.getSuperclass();
        }
      } while (!isFound);
      if (types.length == 0 && !isStatic(modifiers) && isInterface
          && !(Bundle.keywords contains name)) {
        val obj = m.invoke(this)
        obj match {
          case data: Data =>
            setFieldForId(id, data.id, name)
            data.collectElts
            elts += data
          case _ => ()
        }
      }
    }

    elts.sortWith { (a, b) => a.id < b.id }
  }

  override def cloneType: this.type = {
    try {
      val constructor = this.getClass.getConstructors.head
      val res = constructor.newInstance(Array.fill(constructor.getParameterTypes.size)(null):_*)
      val rest = res.asInstanceOf[this.type]
      rest.collectElts
      rest
    } catch {
      case npe: java.lang.reflect.InvocationTargetException if npe.getCause.isInstanceOf[java.lang.NullPointerException] =>
      //   throwException("Parameterized Bundle " + this.getClass + " needs clone method. You are probably using an anonymous Bundle object that captures external state and hence is un-cloneable", npe)
        error("BAD")
      case e: java.lang.Exception =>
        error("BAD")
      //   throwException("Parameterized Bundle " + this.getClass + " needs clone method", e)
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
    m.io.collectElts
    m.setRefs
    val ports = m.io.toPorts
    val component = UniqueComponent(m.name, ports, cmd)
    components += component
    pushCommand(DefInstance(m.defd.id, component.name))
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
  val parent = modulez.headOption
  pushModule(this)
  pushScope
  pushCommands
  addModule(this)

  lazy val params = Module.params
  params.path = this.getClass :: params.path

  def io: Bundle
  def ref = getRefForId(id)
  def lref = ref
  val reset = if (_reset == null) Bool().defd else _reset
  setRefForId(reset.id, "reset")

  def name = {
    // getClass.getName.replace('.', '_')
    getClass.getName.split('.').last
  }
  def debugName: String = {
    val p = parent.getOrElse(null)
    name + (if (p == null) "" else ("__" + p.debugName))
  }

  def setRefs {
    setRefForId(io.id, "this")

    for (m <- getClass.getDeclaredMethods) {
      val name = m.getName()
      val types = m.getParameterTypes()
      if (types.length == 0) {
        val obj = m.invoke(this)
        obj match {
          case module: Module =>
            setRefForId(module.id, name)
            module.setRefs
          case bundle: Bundle =>
            if (name != "io") {
              setRefForId(bundle.id, name)
            }
          case mem: Mem[_] =>
            setRefForId(mem.t.id, name)
          case vec: Vec[_] =>
            setRefForId(vec.id, name)
          case data: Data =>
            setRefForId(data.id, name)
          // ignore anything not of those types
          case _ => null
        }
      }
    }
  }
}

object when {
  def execBlock(block: => Unit): Command = {
    pushScope
    pushCommands
    block
    val cmd = popCommands
    popScope
    cmd
  }
  def execWhen(cond: => Bool)(block: => Unit) {
    val pred = cond.ref
    val cmd  = execBlock(block)
    pushCommand(Conditionally(pred, cmd, EmptyCommand()))
  }
  def apply(cond: => Bool)(block: => Unit): when = {
    execWhen(cond){ block }
    new when
  }
}

import when._

class when {
  def elsewhen (cond: => Bool)(block: => Unit): when = {
    this.otherwise {
      when.execWhen(cond) { block }
    }
    new when
  }

  private def replaceCondition(cond: Conditionally, elsecmd: Command): Conditionally = {
    cond.alt match {
      // this is an elsewhen clause
      // we have to go deeper
      case newcond: Conditionally =>
        Conditionally(cond.pred, cond.conseq,
          replaceCondition(newcond, elsecmd))
      // if the alt is empty, we've found the end
      case empty: EmptyCommand =>
        Conditionally(cond.pred, cond.conseq, elsecmd)
      // this shouldn't happen
      case _ =>
        throw new Exception("Cannot replace non-empty else clause")
    }
  }

  def otherwise (block: => Unit) {
    // first generate the body
    val elsecmd = execBlock(block)
    // now we look back and find the last Conditionally
    val isConditionally = (x: Command) => {
      x match {
        case Conditionally(_, _, _) => true
        case _ => false
      }
    }
    // replace the last Conditionally with a new one with the
    // same predicate and consequent but with the last alt replaced
    // by the commands for the otherwise body
    val i = commands.lastIndexWhere(isConditionally)
    commands(i) = commands(i) match {
      case cond: Conditionally =>
        replaceCondition(cond, elsecmd)
      // this should never happen
      case _ => throw new Exception("That's not a conditionally")
    }
  }
}


/// CHISEL IR EMITTER

class Emitter {
  var indenting = 0
  def withIndent(f: => String) = { 
    indenting += 1;
    val res = f
    indenting -= 1;
    res
  }
  def join(parts: Array[String], sep: String) = 
    parts.foldLeft("")((s, p) => if (s == "") p else s + sep + p)
  def join0(parts: Array[String], sep: String) = 
    parts.foldLeft("")((s, p) => s + sep + p)
  def newline = 
    "\n" + join((0 until indenting).map(x => "  ").toArray, "")
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
      case e: DefPrim => "node " + e.name + " = " + emit(e.op) + "(" + join(e.args.map(x => emit(x)) ++ e.lits.map(x => x.toString), ", ") + ")"
      case e: DefPrimPad => "node " + e.name + " = " + emit(e.op) + "(" + join(e.args.map(x => "Pad(" + emit(x) + ",?)") ++ e.lits.map(x => x.toString), ", ") + ")"
      case e: DefWire => "wire " + e.name + " : " + emitType(e.kind)
      case e: DefRegister => "reg " + e.name + " : " + emitType(e.kind)
      case e: DefMemory => "mem " + e.name + " : " + emitType(e.kind) + "[" + e.size + "]";
      // case e: DefVector => "vec " + e.name + " : " + emit(e.kind) + "(" + join(e.args.map(x => emit(x)).toArray[String], " ") + ")"
      case e: DefAccessor => "accessor " + e.name + " = " + emit(e.source) + "[" + emit(e.index) + "]"
      case e: DefInstance => {
        val mod = modules(e.id)
        // update all references to the modules ports
        setRefForId(mod.io.id, e.name, true)
        "inst " + e.name + " of " + e.module
      }
      case e: Conditionally => "when " + emit(e.pred) + " : " + withIndent{ emit(e.conseq) } + (if (e.alt.isInstanceOf[EmptyCommand]) "" else newline + "else : " + withIndent{ emit(e.alt) })
      case e: Begin => join0(e.body.map(x => emit(x)), newline)
      case e: Connect => emit(e.loc) + " := " + emit(e.exp)
      case e: ConnectPad => emit(e.loc) + " := Pad(" + emit(e.exp) + ",?)"
      case e: ConnectInit => "on-reset " + emit(e.loc) + " := Pad(" + emit(e.exp) + ",?)"
      case e: ConnectInitIndex => "on-reset " + emit(e.loc) + "[" + e.index + "] := Pad(" + emit(e.exp) + ",?)"
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
