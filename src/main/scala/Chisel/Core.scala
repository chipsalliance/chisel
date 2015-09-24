package Chisel
import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashSet, LinkedHashMap}
import Builder.pushCommand
import Builder.pushOp
import Builder.dynamicContext
import PrimOp._

/** A factory for literal values */
private object Literal {
  def sizeof(x: BigInt): Int = x.bitLength

  def decodeBase(base: Char): Int = base match {
    case 'x' | 'h' => 16
    case 'd' => 10
    case 'o' => 8
    case 'b' => 2
    case _ => Builder.error("Invalid base " + base); 2
  }

  def stringToVal(base: Char, x: String): BigInt =
    BigInt(x, decodeBase(base))
}

sealed abstract class Direction(name: String) {
  override def toString = name
  def flip: Direction
}
object INPUT  extends Direction("input") { def flip = OUTPUT }
object OUTPUT extends Direction("output") { def flip = INPUT }
object NO_DIR extends Direction("?") { def flip = NO_DIR }

object debug {
  // TODO:
  def apply (arg: Data) = arg
}

/** *Data* is the root of the type system which includes
  Aggregate (Bundle, Vec) and Element (UInt, SInt, etc.).

  Instances of Data are meant to help with construction and correctness
  of a logic graph. They will trimmed out of the graph before a *Backend*
  generates target code.
  */
abstract class Data(dirArg: Direction) extends HasId {
  def dir: Direction = dirVar

  // Sucks this is mutable state, but cloneType doesn't take a Direction arg
  private var isFlipVar = dirArg == INPUT
  private var dirVar = dirArg
  private[Chisel] def isFlip = isFlipVar

  private def cloneWithDirection(newDir: Direction => Direction,
                                 newFlip: Boolean => Boolean): this.type = {
    val res = this.cloneType
    res.isFlipVar = newFlip(res.isFlipVar)
    for ((me, it) <- this.flatten zip res.flatten)
      (it: Data).dirVar = newDir((me: Data).dirVar)
    res
  }
  def asInput: this.type = cloneWithDirection(_ => INPUT, _ => true)
  def asOutput: this.type = cloneWithDirection(_ => OUTPUT, _ => false)
  def flip(): this.type = cloneWithDirection(_.flip, !_)

  private[Chisel] def badConnect(that: Data): Unit =
    throwException(s"cannot connect ${this} and ${that}")
  private[Chisel] def connect(that: Data): Unit =
    pushCommand(Connect(this.lref, that.ref))
  private[Chisel] def bulkConnect(that: Data): Unit =
    pushCommand(BulkConnect(this.lref, that.lref))
  private[Chisel] def lref: Node = Node(this)
  private[Chisel] def ref: Arg = if (isLit) litArg.get else lref
  private[Chisel] def cloneTypeWidth(width: Width): this.type
  private[Chisel] def toType: String

  def := (that: Data): Unit = this badConnect that
  def <> (that: Data): Unit = this badConnect that
  def cloneType: this.type
  def litArg(): Option[LitArg] = None
  def litValue(): BigInt = litArg.get.num
  def isLit(): Boolean = litArg.isDefined

  def width: Width
  final def getWidth = width.get

  private[Chisel] def flatten: IndexedSeq[Bits]
  def fromBits(n: Bits): this.type = {
    var i = 0
    val wire = Wire(this.cloneType)
    for (x <- wire.flatten) {
      x := n(i + x.getWidth-1, i)
      i += x.getWidth
    }
    wire.asInstanceOf[this.type]
  }
  def toBits(): UInt = Cat(this.flatten.reverse)
}

object Wire {
  def apply[T <: Data](t: T = null, init: T = null): T = {
    val x = Reg.makeType(t, null.asInstanceOf[T], init)
    pushCommand(DefWire(x))
    if (init != null)
      x := init
    else
      x.flatten.foreach(e => e := e.fromInt(0))
    x
  }
}

/** A factory for Reg objects. */
object Reg {
  private[Chisel] def makeType[T <: Data](t: T = null, next: T = null, init: T = null): T = {
    if (t ne null) t.cloneType
    else if (next ne null) next.cloneTypeWidth(Width())
    else if (init ne null) init.litArg match {
      // For e.g. Reg(init=UInt(0, k)), fix the Reg's width to k
      case Some(lit) if lit.forcedWidth => init.cloneType
      case _ => init.cloneTypeWidth(Width())
    } else throwException("cannot infer type")
  }

  def apply[T <: Data](t: T = null, next: T = null, init: T = null): T = {
    val x = makeType(t, next, init)
    pushCommand(DefRegister(x, Node(x._parent.get.clock), Node(x._parent.get.reset))) // TODO multi-clock
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
    pushCommand(DefMemory(mem, mt, size, Node(mt._parent.get.clock))) // TODO multi-clock
    mem
  }
}

sealed abstract class MemBase[T <: Data](t: T, val length: Int) extends HasId with VecLike[T] {
  def apply(idx: Int): T = apply(UInt(idx))
  def apply(idx: UInt): T =
    pushCommand(DefAccessor(t.cloneType, Node(this), NO_DIR, idx.ref)).id

  def read(idx: UInt): T = apply(idx)
  def write(idx: UInt, data: T): Unit = apply(idx) := data
  def write(idx: UInt, data: T, mask: Vec[Bool]) (implicit evidence: T <:< Vec[_]): Unit = {
    val accessor = apply(idx).asInstanceOf[Vec[Data]]
    for (((cond, port), datum) <- mask zip accessor zip data.asInstanceOf[Vec[Data]])
      when (cond) { port := datum }
  }
}

sealed class Mem[T <: Data](t: T, length: Int) extends MemBase(t, length)

object SeqMem {
  def apply[T <: Data](t: T, size: Int): SeqMem[T] = {
    val mt  = t.cloneType
    val mem = new SeqMem(mt, size)
    pushCommand(DefSeqMemory(mem, mt, size, Node(mt._parent.get.clock))) // TODO multi-clock
    mem
  }
}

sealed class SeqMem[T <: Data](t: T, n: Int) extends MemBase[T](t, n) {
  def read(addr: UInt, enable: Bool): T =
    read(Mux(enable, addr, Poison(addr)))
}

object Vec {
  def apply[T <: Data](n: Int, gen: T): Vec[T] = new Vec(gen.cloneType, n)
  
  @deprecated("Chisel3 vec argument order should be n, gen - this will be removed by Chisel3 official release", "now")
  def apply[T <: Data](gen: T, n: Int): Vec[T] = new Vec(gen.cloneType, n)
  /** Returns a new *Vec* from a sequence of *Data* nodes.
    */
  def apply[T <: Data](elts: Seq[T]): Vec[T] = {
    require(!elts.isEmpty)
    val width = elts.map(_.width).reduce(_ max _)
    val vec = new Vec(elts.head.cloneTypeWidth(width), elts.length)
    pushCommand(DefWire(vec))
    for ((v, e) <- vec zip elts)
      v := e
    vec
  }
  /** Returns a new *Vec* from the concatenation of a *Data* node
    and a sequence of *Data* nodes.
    */
  def apply[T <: Data](elt0: T, elts: T*): Vec[T] =
    apply(elt0 +: elts.toSeq)
  /** Returns an array containing values of a given function over
    a range of integer values starting from 0.
    */
  def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] = 
    apply((0 until n).map(i => gen(i)))
  /** Returns an array that contains the results of some element computation
    a number of times.

    Note that this means that elem is computed a total of n times.
    */
  def fill[T <: Data](n: Int)(gen: => T): Vec[T] = apply(Seq.fill(n)(gen))
}

sealed abstract class Aggregate(dirArg: Direction) extends Data(dirArg) {
  private[Chisel] def cloneTypeWidth(width: Width): this.type = cloneType
  def width: Width = flatten.map(_.width).reduce(_ + _)
}

sealed class Vec[T <: Data] private (gen: => T, val length: Int)
    extends Aggregate(gen.dir) with VecLike[T] {
  private val self = IndexedSeq.fill(length)(gen)

  override def <> (that: Data): Unit = that match {
    case _: Vec[_] => this bulkConnect that
    case _ => this badConnect that
  }

  def <> (that: Seq[T]): Unit =
    for ((a, b) <- this zip that)
      a <> b

  def <> (that: Vec[T]): Unit = this bulkConnect that

  override def := (that: Data): Unit = that match {
    case _: Vec[_] => this connect that
    case _ => this badConnect that
  }

  def := (that: Seq[T]): Unit = {
    require(this.length == that.length)
    for ((a, b) <- this zip that)
      a := b
  }

  def := (that: Vec[T]): Unit = this connect that

  def apply(idx: UInt): T = {
    val x = gen
    pushCommand(DefAccessor(x, Node(this), NO_DIR, idx.ref))
    x
  }

  def apply(idx: Int): T = self(idx)
  def read(idx: UInt): T = apply(idx)
  def write(idx: UInt, data: T): Unit = apply(idx) := data

  override def cloneType: this.type =
    Vec(gen, length).asInstanceOf[this.type]

  private val t = gen
  private[Chisel] def toType: String = s"${t.toType}[$length]"
  private[Chisel] lazy val flatten: IndexedSeq[Bits] =
    (0 until length).flatMap(i => this.apply(i).flatten)

  for ((elt, i) <- self zipWithIndex)
    elt.setRef(this, i)
}

trait VecLike[T <: Data] extends collection.IndexedSeq[T] {
  def read(idx: UInt): T
  def write(idx: UInt, data: T): Unit
  def apply(idx: UInt): T

  def forall(p: T => Bool): Bool = (this map p).fold(Bool(true))(_&&_)
  def exists(p: T => Bool): Bool = (this map p).fold(Bool(false))(_||_)
  def contains(x: T) (implicit evidence: T <:< UInt): Bool = this.exists(_ === x)
  def count(p: T => Bool): UInt = PopCount((this map p).toSeq)

  private def indexWhereHelper(p: T => Bool) = this map p zip (0 until length).map(i => UInt(i))
  def indexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p))
  def lastIndexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p).reverse)
  def onlyIndexWhere(p: T => Bool): UInt = Mux1H(indexWhereHelper(p))
}

/** A bit pattern object to enable representation of dont cares */
object BitPat {
  private def parse(x: String): (BigInt, BigInt, Int) = {
    require(x.head == 'b', "BINARY BitPats ONLY")
    var bits = BigInt(0)
    var mask = BigInt(0)
    for (d <- x.tail) {
      if (d != '_') {
        if (!"01?".contains(d)) Builder.error({"Literal: " + x + " contains illegal character: " + d})
        mask = (mask << 1) + (if (d == '?') 0 else 1)
        bits = (bits << 1) + (if (d == '1') 1 else 0)
      }
    }
    (bits, mask, x.length-1)
  }

  /** Get a bit pattern from a string
    * @param n a string with format b---- eg) b1?01
    * @note legal characters are 0, 1, ? and must be base 2*/
  def apply(n: String): BitPat = {
    val (bits, mask, width) = parse(n)
    new BitPat(bits, mask, width)
  }

  /** Get a bit pattern of don't cares with a specified width */
  def DC(width: Int): BitPat = BitPat("b" + ("?" * width))

  // BitPat <-> UInt
  /** enable conversion of a bit pattern to a UInt */
  implicit def BitPatToUInt(x: BitPat): UInt = {
    require(x.mask == (BigInt(1) << x.getWidth)-1)
    UInt(x.value, x.getWidth)
  }
  /** create a bit pattern from a UInt */
  implicit def apply(x: UInt): BitPat = {
    require(x.isLit)
    BitPat("b" + x.litValue.toString(2))
  }
}

/** A class to create bit patterns
  * Use the [[Chisel.BitPat$ BitPat]] object instead of this class directly */
sealed class BitPat(val value: BigInt, val mask: BigInt, width: Int) {
  def getWidth: Int = width
  def === (other: UInt): Bool = UInt(value) === (other & UInt(mask))
  def != (other: UInt): Bool = !(this === other)
}

abstract class Element(dirArg: Direction, val width: Width) extends Data(dirArg) {
  private[Chisel] def flatten: IndexedSeq[UInt] = IndexedSeq(toBits)
}

/** Create a new Clock */
object Clock {
  def apply(dir: Direction = NO_DIR): Clock = new Clock(dir)
}

sealed class Clock(dirArg: Direction) extends Element(dirArg, Width(1)) {
  def cloneType: this.type = Clock(dirArg).asInstanceOf[this.type]
  private[Chisel] override def flatten: IndexedSeq[UInt] = IndexedSeq()
  private[Chisel] def cloneTypeWidth(width: Width): this.type = cloneType
  private[Chisel] def toType = "Clock"

  override def := (that: Data): Unit = that match {
    case _: Clock => this connect that
    case _ => this badConnect that
  }
}

sealed abstract class Bits(dirArg: Direction, width: Width, override val litArg: Option[LitArg]) extends Element(dirArg, width) {
  def fromInt(x: BigInt): this.type
  def makeLit(value: BigInt): LitArg
  def cloneType: this.type = cloneTypeWidth(width)

  override def <> (that: Data): Unit = this := that

  /** Extract a single Bool at index *bit*.
    */
  final def apply(x: BigInt): Bool = {
    if (x < 0)
      Builder.error(s"Negative bit indices are illegal (got $x)")
    if (isLit()) Bool(((litValue() >> x.toInt) & 1) == 1)
    else pushOp(DefPrim(Bool(), BitSelectOp, this.ref, ILit(x)))
  }
  final def apply(x: Int): Bool =
    apply(BigInt(x))
  final def apply(x: UInt): Bool =
    (this >> x)(0)

  /** Extract a range of bits, inclusive from hi to lo
    * {{{ myBits = 0x5, myBits(1,0) => 0x1 }}} */
  final def apply(x: Int, y: Int): UInt = {
    if (x < y || y < 0)
      Builder.error(s"Invalid bit range ($x,$y)")
    val w = x - y + 1
    if (isLit()) UInt((litValue >> y) & ((BigInt(1) << w) - 1), w)
    else pushOp(DefPrim(UInt(width = w), BitsExtractOp, this.ref, ILit(x), ILit(y)))
  }
  final def apply(x: BigInt, y: BigInt): UInt = apply(x.toInt, y.toInt)

  private[Chisel] def unop[T <: Data](dest: T, op: PrimOp): T =
    pushOp(DefPrim(dest, op, this.ref))
  private[Chisel] def binop[T <: Data](dest: T, op: PrimOp, other: BigInt): T =
    pushOp(DefPrim(dest, op, this.ref, ILit(other)))
  private[Chisel] def binop[T <: Data](dest: T, op: PrimOp, other: Bits): T =
    pushOp(DefPrim(dest, op, this.ref, other.ref))
  private[Chisel] def compop(op: PrimOp, other: Bits): Bool =
    pushOp(DefPrim(Bool(), op, this.ref, other.ref))
  private[Chisel] def redop(op: PrimOp): Bool =
    pushOp(DefPrim(Bool(), op, this.ref))

  /** invert all bits with ~ */
  def unary_~ : this.type = unop(cloneTypeWidth(width), BitNotOp)
  def pad (other: Int): this.type = binop(cloneTypeWidth(this.width max Width(other)), PadOp, other)

  /** Shift left operation */
  def << (other: BigInt): Bits
  def << (other: Int): Bits
  def << (other: UInt): Bits
  /** Shift right operation */
  def >> (other: BigInt): Bits
  def >> (other: Int): Bits
  def >> (other: UInt): Bits

  /** Split up this bits instantiation to a Vec of Bools */
  def toBools: Vec[Bool] = Vec.tabulate(this.getWidth)(i => this(i))

  def asSInt(): SInt
  def asUInt(): UInt
  final def toSInt(): SInt = asSInt
  final def toUInt(): UInt = asUInt

  def toBool(): Bool = width match {
    case KnownWidth(1) => this(0)
    case _ => throwException(s"can't covert UInt<$width> to Bool")
  }

  /** Cat bits together to into a single data object with the width of both combined */
  def ## (other: Bits): UInt = Cat(this, other)
  override def toBits = asUInt
  override def fromBits(n: Bits): this.type = {
    val res = Wire(this).asInstanceOf[this.type]
    res := n
    res
  }
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

sealed class UInt private[Chisel] (dir: Direction, width: Width, lit: Option[ULit] = None) extends Bits(dir, width, lit) with Num[UInt] {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type =
    new UInt(dir, w).asInstanceOf[this.type]
  private[Chisel] def toType = s"UInt<$width>"

  def fromInt(value: BigInt): this.type = UInt(value).asInstanceOf[this.type]
  def makeLit(value: BigInt): ULit = ULit(value, Width())

  override def := (that: Data): Unit = that match {
    case _: UInt => this connect that
    case _ => this badConnect that
  }

  def unary_- = UInt(0) - this
  def unary_-% = UInt(0) -% this
  def +& (other: UInt): UInt = binop(UInt((this.width max other.width) + 1), AddOp, other)
  def + (other: UInt): UInt = this +% other
  def +% (other: UInt): UInt = binop(UInt(this.width max other.width), AddModOp, other)
  def -& (other: UInt): UInt = binop(UInt((this.width max other.width) + 1), SubOp, other)
  def - (other: UInt): UInt = this -% other
  def -% (other: UInt): UInt = binop(UInt(this.width max other.width), SubModOp, other)
  def * (other: UInt): UInt = binop(UInt(this.width + other.width), TimesOp, other)
  def * (other: SInt): SInt = other * this
  def / (other: UInt): UInt = binop(UInt(this.width), DivideOp, other)
  def % (other: UInt): UInt = binop(UInt(this.width), ModOp, other)

  def & (other: UInt): UInt = binop(UInt(this.width max other.width), BitAndOp, other)
  def | (other: UInt): UInt = binop(UInt(this.width max other.width), BitOrOp, other)
  def ^ (other: UInt): UInt = binop(UInt(this.width max other.width), BitXorOp, other)

  def orR = this != UInt(0)
  def andR = ~this === UInt(0)
  def xorR = redop(XorReduceOp)

  def < (other: UInt): Bool = compop(LessOp, other)
  def > (other: UInt): Bool = compop(GreaterOp, other)
  def <= (other: UInt): Bool = compop(LessEqOp, other)
  def >= (other: UInt): Bool = compop(GreaterEqOp, other)
  def != (other: UInt): Bool = compop(NotEqualOp, other)
  def === (other: UInt): Bool = compop(EqualOp, other)
  def unary_! : Bool = this === Bits(0)

  def << (other: Int): UInt = binop(UInt(this.width + other), ShiftLeftOp, other)
  def << (other: BigInt): UInt = this << other.toInt
  def << (other: UInt): UInt = binop(UInt(this.width.dynamicShiftLeft(other.width)), DynamicShiftLeftOp, other)
  def >> (other: Int): UInt = binop(UInt(this.width.shiftRight(other)), ShiftRightOp, other)
  def >> (other: BigInt): UInt = this >> other.toInt
  def >> (other: UInt): UInt = binop(UInt(this.width), DynamicShiftRightOp, other)

  def bitSet(off: UInt, dat: Bool): UInt = {
    val bit = UInt(1, 1) << off
    Mux(dat, this | bit, ~(~this | bit))
  }

  def === (that: BitPat): Bool = that === this
  def != (that: BitPat): Bool = that != this

  /** Convert a UInt to an SInt by added a MSB zero */
  def zext(): SInt = pushOp(DefPrim(SInt(width + 1), ConvertOp, ref))
  def asSInt(): SInt = pushOp(DefPrim(SInt(width), AsSIntOp, ref))
  def asUInt(): UInt = this
}

sealed trait UIntFactory {
  /** Create a UInt type with inferred width. */
  def apply(): UInt = apply(NO_DIR, Width())
  /** Create a UInt type or port with fixed width. */
  def apply(dir: Direction = NO_DIR, width: Int): UInt = apply(dir, Width(width))
  /** Create a UInt port with inferred width. */
  def apply(dir: Direction): UInt = apply(dir, Width())

  /** Create a UInt literal with inferred width. */
  def apply(value: BigInt): UInt = apply(value, Width())
  /** Create a UInt literal with fixed width. */
  def apply(value: BigInt, width: Int): UInt = apply(value, Width(width))
  /** Create a UInt literal with inferred width. */
  def apply(n: String): UInt = apply(parse(n), parsedWidth(n))
  /** Create a UInt literal with fixed width. */
  def apply(n: String, width: Int): UInt = apply(parse(n), width)

  /** Create a UInt type with specified width. */
  def apply(width: Width): UInt = apply(NO_DIR, width)
  /** Create a UInt port with specified width. */
  def apply(dir: Direction, width: Width): UInt = new UInt(dir, width)
  /** Create a UInt literal with specified width. */
  def apply(value: BigInt, width: Width): UInt = {
    val lit = ULit(value, width)
    new UInt(NO_DIR, lit.width, Some(lit))
  }

  private def parse(n: String) =
    Literal.stringToVal(n(0), n.substring(1, n.length))
  private def parsedWidth(n: String) =
    if (n(0) == 'b') Width(n.length-1)
    else if (n(0) == 'h') Width((n.length-1) * 4)
    else Width()
}

/** Provides a set of operations to create UInt types and literals. */
object UInt extends UIntFactory

/** Provides a set of operations to create UInt types and literals.
  * Identical in functionality to the UInt companion object. */
object Bits extends UIntFactory

sealed class SInt private (dir: Direction, width: Width, lit: Option[SLit] = None) extends Bits(dir, width, lit) with Num[SInt] {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type =
    new SInt(dir, w).asInstanceOf[this.type]
  private[Chisel] def toType = s"SInt<$width>"

  override def := (that: Data): Unit = that match {
    case _: SInt => this connect that
    case _ => this badConnect that
  }

  def fromInt(value: BigInt): this.type = SInt(value).asInstanceOf[this.type]
  def makeLit(value: BigInt): SLit = SLit(value, Width())

  def unary_- : SInt = SInt(0) - this
  def unary_-% : SInt = SInt(0) -% this
  /** add (width +1) operator */
  def +& (other: SInt): SInt = binop(SInt((this.width max other.width) + 1), AddOp, other)
  /** add (default - no growth) operator */
  def + (other: SInt): SInt = this +% other
  /** add (no growth) operator */
  def +% (other: SInt): SInt = binop(SInt(this.width max other.width), AddModOp, other)
  /** subtract (width +1) operator */
  def -& (other: SInt): SInt = binop(SInt((this.width max other.width) + 1), SubOp, other)
  /** subtract (default - no growth) operator */
  def - (other: SInt): SInt = this -% other
  /** subtract (no growth) operator */
  def -% (other: SInt): SInt = binop(SInt(this.width max other.width), SubModOp, other)
  def * (other: SInt): SInt = binop(SInt(this.width + other.width), TimesOp, other)
  def * (other: UInt): SInt = binop(SInt(this.width + other.width), TimesOp, other)
  def / (other: SInt): SInt = binop(SInt(this.width), DivideOp, other)
  def % (other: SInt): SInt = binop(SInt(this.width), ModOp, other)

  def & (other: SInt): SInt = binop(SInt(this.width max other.width), BitAndOp, other)
  def | (other: SInt): SInt = binop(SInt(this.width max other.width), BitOrOp, other)
  def ^ (other: SInt): SInt = binop(SInt(this.width max other.width), BitXorOp, other)

  def < (other: SInt): Bool = compop(LessOp, other)
  def > (other: SInt): Bool = compop(GreaterOp, other)
  def <= (other: SInt): Bool = compop(LessEqOp, other)
  def >= (other: SInt): Bool = compop(GreaterEqOp, other)
  def != (other: SInt): Bool = compop(NotEqualOp, other)
  def === (other: SInt): Bool = compop(EqualOp, other)
  def abs(): UInt = Mux(this < SInt(0), (-this).toUInt, this.toUInt)

  def << (other: Int): SInt = binop(SInt(this.width + other), ShiftLeftOp, other)
  def << (other: BigInt): SInt = this << other.toInt
  def << (other: UInt): SInt = binop(SInt(this.width.dynamicShiftLeft(other.width)), DynamicShiftLeftOp, other)
  def >> (other: Int): SInt = binop(SInt(this.width.shiftRight(other)), ShiftRightOp, other)
  def >> (other: BigInt): SInt = this >> other.toInt
  def >> (other: UInt): SInt = binop(SInt(this.width), DynamicShiftRightOp, other)

  def asUInt(): UInt = pushOp(DefPrim(UInt(this.width), AsUIntOp, ref))
  def asSInt(): SInt = this
}

/** Provides a set of operations to create SInt types and literals. */
object SInt {
  /** Create an SInt type with inferred width. */
  def apply(): SInt = apply(NO_DIR, Width())
  /** Create an SInt type or port with fixed width. */
  def apply(dir: Direction = NO_DIR, width: Int): SInt = apply(dir, Width(width))
  /** Create an SInt port with inferred width. */
  def apply(dir: Direction): SInt = apply(dir, Width())

  /** Create an SInt literal with inferred width. */
  def apply(value: BigInt): SInt = apply(value, Width())
  /** Create an SInt literal with fixed width. */
  def apply(value: BigInt, width: Int): SInt = apply(value, Width(width))

  /** Create an SInt type with specified width. */
  def apply(width: Width): SInt = new SInt(NO_DIR, width)
  /** Create an SInt port with specified width. */
  def apply(dir: Direction, width: Width): SInt = new SInt(dir, width)
  /** Create an SInt literal with specified width. */
  def apply(value: BigInt, width: Width): SInt = {
    val lit = SLit(value, width)
    new SInt(NO_DIR, lit.width, Some(lit))
  }
}

sealed class Bool(dir: Direction, lit: Option[ULit] = None) extends UInt(dir, Width(1), lit) {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type = {
    require(!w.known || w.get == 1)
    new Bool(dir).asInstanceOf[this.type]
  }

  override def fromInt(value: BigInt): this.type = {
    require(value == 0 || value == 1)
    Bool(value == 1).asInstanceOf[this.type]
  }

  def & (other: Bool): Bool = binop(Bool(), BitAndOp, other)
  def | (other: Bool): Bool = binop(Bool(), BitOrOp, other)
  def ^ (other: Bool): Bool = binop(Bool(), BitXorOp, other)

  def || (that: Bool): Bool = this | that
  def && (that: Bool): Bool = this & that
}

object Bool {
  def apply(dir: Direction = NO_DIR): Bool = new Bool(dir)
  def apply(x: Boolean): Bool = new Bool(NO_DIR, Some(ULit(if (x) 1 else 0, Width(1))))
}

object Mux {
  /** Create a Mux
    * @param cond a condition to determine which value to choose
    * @param con the value chosen when cond is true or 1
    * @param alt the value chosen when cond is false or 0
    * @example
    * {{{ val muxOut = Mux(data_in === UInt(3), UInt(3, 4), UInt(0, 4)) }}} */
  def apply[T <: Data](cond: Bool, con: T, alt: T): T = (con, alt) match {
    // Handle Mux(cond, UInt, Bool) carefully so that the concrete type is UInt
    case (c: Bool, a: Bool) => doMux(cond, c, a).asInstanceOf[T]
    case (c: UInt, a: Bool) => doMux(cond, c, a << 0).asInstanceOf[T]
    case (c: Bool, a: UInt) => doMux(cond, c << 0, a).asInstanceOf[T]
    case (c: Bits, a: Bits) => doMux(cond, c, a).asInstanceOf[T]
    // FIRRTL doesn't support Mux for aggregates, so use a when instead
    case _ => doWhen(cond, con, alt)
  }

  private def doMux[T <: Bits](cond: Bool, con: T, alt: T): T = {
    require(con.getClass == alt.getClass, s"can't Mux between ${con.getClass} and ${alt.getClass}")
    val d = alt.cloneTypeWidth(con.width max alt.width)
    pushOp(DefPrim(d, MultiplexOp, cond.ref, con.ref, alt.ref))
  }
  // This returns an lvalue, which it most definitely should not
  private def doWhen[T <: Data](cond: Bool, con: T, alt: T): T = {
    require(con.getClass == alt.getClass, s"can't Mux between ${con.getClass} and ${alt.getClass}")
    val res = Wire(t = alt.cloneTypeWidth(con.width max alt.width), init = alt)
    when (cond) { res := con }
    res
  }
}

object Cat {
  /** Combine data elements together
    * @param a Data to combine with
    * @param r any number of other Data elements to be combined in order
    * @return A UInt which is all of the bits combined together
    */
  def apply[T <: Bits](a: T, r: T*): UInt = apply(a :: r.toList)
  /** Combine data elements together
    * @param r any number of other Data elements to be combined in order
    * @return A UInt which is all of the bits combined together
    */
  def apply[T <: Bits](r: Seq[T]): UInt = {
    if (r.tail.isEmpty) r.head.asUInt
    else {
      val left = apply(r.slice(0, r.length/2))
      val right = apply(r.slice(r.length/2, r.length))
      val w = left.width + right.width
      pushOp(DefPrim(UInt(w), ConcatOp, left.ref, right.ref))
    }
  }
}

object Bundle {
  private val keywords =
    HashSet[String]("flip", "asInput", "asOutput", "cloneType", "toBits")
  def apply[T <: Bundle](b: => T)(implicit p: Parameters): T = {
    Builder.paramsScope(p.push){ b }
  }
  //TODO @deprecated("Use Chisel.paramsScope object","08-01-2015")
  def apply[T <: Bundle](b: => T,  f: PartialFunction[Any,Any]): T = {
    val q = Builder.getParams.alterPartial(f)
    apply(b)(q)
  }
}

/** Define a collection of datum of different types into a single coherent
  whole.
  */
class Bundle extends Aggregate(NO_DIR) {
  private val _namespace = Builder.globalNamespace.child

  /** Connect all elements contained in this to node 'that'
    * @note The elements are checked for compatibility based on their name
    * If elements are in src that are not in this Bundle no warning will be produced
    * @example
    * {{{ // pass through all wires in this modules io to the sub module which have the same name
    * // Note: ignores any extra defined in io
    * mySubModule.io <> io }}}
    */
  override def <> (that: Data): Unit = that match {
    case _: Bundle => this bulkConnect that
    case _ => this badConnect that
  }

  override def := (that: Data): Unit = this <> that

  lazy val elements: ListMap[String, Data] = ListMap(namedElts:_*)

  private def isBundleField(m: java.lang.reflect.Method) =
    m.getParameterTypes.isEmpty &&
    !java.lang.reflect.Modifier.isStatic(m.getModifiers) &&
    classOf[Data].isAssignableFrom(m.getReturnType) &&
    !(Bundle.keywords contains m.getName) && !(m.getName contains '$')

  private[Chisel] lazy val namedElts = {
    val nameMap = LinkedHashMap[String, Data]()
    val seen = HashSet[Data]()
    for (m <- getClass.getMethods.sortWith(_.getName < _.getName); if isBundleField(m)) {
      m.invoke(this) match {
        case d: Data =>
          if (nameMap contains m.getName) require(nameMap(m.getName) eq d)
          else if (!seen(d)) { nameMap(m.getName) = d; seen += d }
        case _ =>
      }
    }
    ArrayBuffer(nameMap.toSeq:_*) sortWith {case ((an, a), (bn, b)) => (a._id > b._id) || ((a eq b) && (an > bn))}
  }
  private[Chisel] def toType = s"{${namedElts.reverse.map(e => (if (e._2.isFlip) "flip " else "")+e._2.getRef.name+" : "+e._2.toType).reduce(_+", "+_)}}"
  private[Chisel] lazy val flatten = namedElts.flatMap(_._2.flatten)
  private[Chisel] def addElt(name: String, elt: Data): Unit =
    namedElts += name -> elt
  private[Chisel] override def _onModuleClose: Unit =
    for ((name, elt) <- namedElts) { elt.setRef(this, _namespace.name(name)) }

  override def cloneType : this.type = {
    try {
      val constructor = this.getClass.getConstructors.head
      val res = constructor.newInstance(Seq.fill(constructor.getParameterTypes.size)(null):_*)
      res.asInstanceOf[this.type]
    } catch {
      case npe: java.lang.reflect.InvocationTargetException if npe.getCause.isInstanceOf[java.lang.NullPointerException] =>
        Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method. You are probably using an anonymous Bundle object that captures external state and hence is un-cloneTypeable")
        this
      case npe: java.lang.reflect.InvocationTargetException =>
        Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method")
        this
    }
  }
}

object Module {
  /** The wrapper method for all instantiations of modules
    * @param m The module newly created
    * @param p Parameters passed down implicitly from that it is created in
    * @return A properly wrapped module that has been added to the Driver
    */
  def apply[T <: Module](bc: => T)(implicit currParams: Parameters = Builder.getParams.push): T = {
    paramsScope(currParams) {
      val parent = dynamicContext.currentModule
      val m = bc.setRefs()
      // init module outputs
      m._commands prependAll (for (p <- m.io.flatten; if p.dir == OUTPUT)
        yield Connect(p.lref, p.fromInt(0).ref))
      dynamicContext.currentModule = parent
      val ports = m.computePorts
      Builder.components += Component(m, m.name, ports, m._commands)
      pushCommand(DefInstance(m, ports))
      // init instance inputs
      for (p <- m.io.flatten; if p.dir == INPUT)
        p := p.fromInt(0)
      m
    }.connectImplicitIOs()
  }
  //TODO @deprecated("Use Chisel.paramsScope object","08-01-2015")
  def apply[T <: Module](m: => T, f: PartialFunction[Any,Any]): T = {
    apply(m)(Builder.getParams.alterPartial(f))
  }
}

abstract class Module(_clock: Clock = null, _reset: Bool = null) extends HasId {
  private val _namespace = Builder.globalNamespace.child
  private[Chisel] val _commands = ArrayBuffer[Command]()
  private[Chisel] val _ids = ArrayBuffer[HasId]()
  dynamicContext.currentModule = Some(this)

  /** Name of the instance. */
  val name = Builder.globalNamespace.name(getClass.getName.split('.').last)

  /** the I/O for this module */
  def io: Bundle
  val clock = Clock(INPUT)
  val reset = Bool(INPUT)

  private[Chisel] def addId(d: HasId) { _ids += d }
  private[Chisel] def ref = Builder.globalRefMap(this)
  private[Chisel] def lref = ref

  private def ports = (clock, "clock") :: (reset, "reset") :: (io, "io") :: Nil

  private[Chisel] def computePorts = ports map { case (port, name) =>
    val bundleDir = if (port.isFlip) INPUT else OUTPUT
    Port(port, if (port.dir == NO_DIR) bundleDir else port.dir)
  }

  private def connectImplicitIOs(): this.type = _parent match {
    case Some(p) =>
      clock := (if (_clock eq null) p.clock else _clock)
      reset := (if (_reset eq null) p.reset else _reset)
      this
    case None => this
  }

  private def makeImplicitIOs(): Unit = ports map { case (port, name) =>
  }

  private def setRefs(): this.type = {
    for ((port, name) <- ports)
      port.setRef(ModuleIO(this, _namespace.name(name)))

    val valNames = HashSet[String](getClass.getDeclaredFields.map(_.getName):_*)
    def isPublicVal(m: java.lang.reflect.Method) =
      m.getParameterTypes.isEmpty && valNames.contains(m.getName)
    val methods = getClass.getMethods.sortWith(_.getName > _.getName)
    for (m <- methods; if isPublicVal(m)) m.invoke(this) match {
      case id: HasId => id.setRef(_namespace.name(m.getName))
      case _ =>
    }

    _ids.foreach(_.setRef(_namespace.name("T")))
    _ids.foreach(_._onModuleClose)
    this
  }

  // TODO: actually implement these
  def assert(cond: Bool, msg: String): Unit = {}
  def printf(message: String, args: Bits*): Unit = {}
}

/** This class allows the connection to modules defined outside of chisel.
  * @example
  * {{{ class DSP48E1 extends BlackBox {
  *      val io = new Bundle // Create I/O with same as DSP
  *      val dspParams = new VerilogParameters // Create Parameters to be specified
  *      setVerilogParams(dspParams)
  *      // Implement functionality of DSP to allow simulation verification
  * } }}}
  */
// TODO: actually implement BlackBox (this hack just allows them to compile)
abstract class BlackBox(_clock: Clock = null, _reset: Bool = null) extends Module(_clock = _clock, _reset = _reset) {
  def setVerilogParameters(s: String): Unit = {}
}

/** An object to create conditional logic.
  * @example
  * {{{
  * when ( myData === UInt(3) ) {
  *   ... // Some logic
  * } .elsewhen ( myData === UInt(1) ) {
  *   ... // Some logic
  * } .otherwise {
  *   ... // Some logic
  * } }}}
  */
object when {
  /** @param cond condition to execute upon
    * @param block a section of logic to enable if cond is true */
  def apply(cond: => Bool)(block: => Unit): WhenContext = {
    new WhenContext(cond)(block)
  }
}

class WhenContext(cond: => Bool)(block: => Unit) {
  /** execute block when alternative cond is true */
  def elsewhen (cond: => Bool)(block: => Unit): WhenContext =
    doOtherwise(when(cond)(block))

  /** execute block by default */
  def otherwise(block: => Unit): Unit =
    doOtherwise(block)

  pushCommand(WhenBegin(cond.ref))
  block
  pushCommand(WhenEnd())

  private def doOtherwise[T](block: => T): T = {
    pushCommand(WhenElse())
    val res = block
    pushCommand(WhenEnd())
    res
  }
}

/** A source of garbage data, used to initialize Wires to a don't-care value. */
private object Poison extends Command {
  def apply[T <: Data](t: T): T =
    pushCommand(DefPoison(t.cloneType)).id
}
