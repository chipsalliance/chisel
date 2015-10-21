// See LICENSE for license details.

package Chisel
import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashSet, LinkedHashMap}
import Builder.pushCommand
import Builder.pushOp
import Builder.dynamicContext
import PrimOp._

sealed abstract class Direction(name: String) {
  override def toString: String = name
  def flip: Direction
}
object INPUT  extends Direction("input") { override def flip: Direction = OUTPUT }
object OUTPUT extends Direction("output") { override def flip: Direction = INPUT }
object NO_DIR extends Direction("?") { override def flip: Direction = NO_DIR }

// REVIEW TODO: Should this actually be part of the RTL API? RTL should be
// considered untouchable from a debugging standpoint?
object debug {
  // TODO:
  def apply (arg: Data): Data = arg
}

/** This forms the root of the type system for wire data types. The data value
  * must be representable as some number (need not be known at Chisel compile
  * time) of bits, and must have methods to pack / unpack structured data to /
  * from bits.
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

  // REVIEW TODO: Can these just be abstract, and left to implementing classes
  // to define them (or even undefined)? Bonus: compiler can help you catch
  // unimplemented functions.
  def := (that: Data): Unit = this badConnect that
  def <> (that: Data): Unit = this badConnect that
  def cloneType: this.type
  def litArg(): Option[LitArg] = None
  def litValue(): BigInt = litArg.get.num
  def isLit(): Boolean = litArg.isDefined

  def width: Width
  final def getWidth: Int = width.get

  // REVIEW TODO: should this actually be part of the Data interface? this is
  // an Aggregate function?
  private[Chisel] def flatten: IndexedSeq[Bits]

  /** Creates an new instance of this type, unpacking the input Bits into
    * structured data. Generates no logic (should be either wires or a syntactic
    * transformation).
    *
    * This performs the inverse operation of toBits.
    *
    * @note does NOT assign to the object this is called on, instead creating a
    * NEW object
    */
  def fromBits(n: Bits): this.type = {
    // REVIEW TODO: width match checking?
    // REVIEW TODO: perhaps have a assign version, especially since this is
    // called from a specific object, instead of a factory constructor. It's
    // not immediately obvious that this creates a new object.
    var i = 0
    val wire = Wire(this.cloneType)
    for (x <- wire.flatten) {
      x := n(i + x.getWidth-1, i)
      i += x.getWidth
    }
    wire.asInstanceOf[this.type]
  }

  /** Packs the value of this object as plain Bits. Generates no logic (should
    * be either wires or a syntactic transformation).
    *
    * This performs the inverse operation of fromBits(Bits).
    */
  def toBits(): UInt = Cat(this.flatten.reverse)
}

object Wire {
  def apply[T <: Data](t: T = null, init: T = null): T = {
    val x = Reg.makeType(t, null.asInstanceOf[T], init)
    pushCommand(DefWire(x))
    if (init != null) {
      x := init
    } else {
      x.flatten.foreach(e => e := e.fromInt(0))
    }
    x
  }
}

object Reg {
  private[Chisel] def makeType[T <: Data](t: T = null, next: T = null, init: T = null): T = {
    if (t ne null) {
      t.cloneType
    } else if (next ne null) {
      next.cloneTypeWidth(Width())
    } else if (init ne null) {
      init.litArg match {
        // For e.g. Reg(init=UInt(0, k)), fix the Reg's width to k
        case Some(lit) if lit.forcedWidth => init.cloneType
        case _ => init.cloneTypeWidth(Width())
      }
    } else {
      throwException("cannot infer type")
    }
  }

  /** Creates a register with optional next and initialization values.
    *
    * @param t: data type for the register
    * @param next: new value register is to be updated with every cycle (or
    * empty to not update unless assigned to using the := operator)
    * @param init: initialization value on reset (or empty for uninitialized,
    * where the register value persists across a reset)
    */
  def apply[T <: Data](t: T = null, next: T = null, init: T = null): T = {
    // REVIEW TODO: rewrite this in a less brittle way, perhaps also in a way
    // that doesn't need two implementations of apply()
    val x = makeType(t, next, init)
    pushCommand(DefRegister(x, Node(x._parent.get.clock), Node(x._parent.get.reset))) // TODO multi-clock
    if (init != null) {
      pushCommand(ConnectInit(x.lref, init.ref))
    }
    if (next != null) {
      x := next
    }
    x
  }

  /** Creates a register without initialization (reset is ignored). Value does
    * not change unless assigned to (using the := operator).
    *
    * @param outType: data type for the register
    */
  def apply[T <: Data](outType: T): T = Reg[T](outType, null.asInstanceOf[T], null.asInstanceOf[T])
}

object Mem {
  @deprecated("Mem argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](t: T, size: Int): Mem[T] = apply(size, t)

  /** Creates a combinational-read, sequential-write [[Mem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): Mem[T] = {
    val mt  = t.cloneType
    val mem = new Mem(mt, size)
    pushCommand(DefMemory(mem, mt, size, Node(mt._parent.get.clock))) // TODO multi-clock
    mem
  }
}

sealed abstract class MemBase[T <: Data](t: T, val length: Int) extends HasId with VecLike[T] {
  // REVIEW TODO: make accessors (static/dynamic, read/write) combinations consistent.

  /** Creates a read accessor into the memory with static addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: Int): T = apply(UInt(idx))

  /** Creates a read accessor into the memory with dynamic addressing. See the
    * class documentation of the memory for more detailed information.
    */
  def apply(idx: UInt): T =
    pushCommand(DefAccessor(t.cloneType, Node(this), NO_DIR, idx.ref)).id

  def read(idx: UInt): T = apply(idx)

  /** Creates a write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    */
  def write(idx: UInt, data: T): Unit = apply(idx) := data

  /** Creates a masked write accessor into the memory.
    *
    * @param idx memory element index to write into
    * @param data new data to write
    * @param mask write mask as a Vec of Bool: a write to the Vec element in
    * memory is only performed if the corresponding mask index is true.
    *
    * @note this is only allowed if the memory's element data type is a Vec
    */
  def write(idx: UInt, data: T, mask: Vec[Bool]) (implicit evidence: T <:< Vec[_]): Unit = {
    // REVIEW TODO: error checking to detect zip length mismatch?

    val accessor = apply(idx).asInstanceOf[Vec[Data]]
    for (((cond, port), datum) <- mask zip accessor zip data.asInstanceOf[Vec[Data]])
      when (cond) { port := datum }
  }
}

/** A combinational-read, sequential-write memory.
  *
  * Writes take effect on the rising clock edge after the request. Reads are
  * combinational (requests will return data on the same cycle).
  * Read-after-write hazards are not an issue.
  */
sealed class Mem[T <: Data](t: T, length: Int) extends MemBase(t, length)

object SeqMem {
  @deprecated("SeqMem argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](t: T, size: Int): SeqMem[T] = apply(size, t)

  /** Creates a sequential-read, sequential-write [[SeqMem]].
    *
    * @param size number of elements in the memory
    * @param t data type of memory element
    */
  def apply[T <: Data](size: Int, t: T): SeqMem[T] = {
    val mt  = t.cloneType
    val mem = new SeqMem(mt, size)
    pushCommand(DefSeqMemory(mem, mt, size, Node(mt._parent.get.clock))) // TODO multi-clock
    mem
  }
}

/** A sequential-read, sequential-write memory.
  *
  * Writes take effect on the rising clock edge after the request. Reads return
  * data on the rising edge after the request. Read-after-write behavior (when
  * a read and write to the same address are requested on the same cycle) is
  * undefined.
  */
sealed class SeqMem[T <: Data](t: T, n: Int) extends MemBase[T](t, n) {
  def read(addr: UInt, enable: Bool): T =
    read(Mux(enable, addr, Poison(addr)))
}

object Vec {
  /** Creates a new [[Vec]] with `n` entries of the specified data type.
    *
    * @note elements are NOT assigned by default and have no value
    */
  def apply[T <: Data](n: Int, gen: T): Vec[T] = new Vec(gen.cloneType, n)

  @deprecated("Vec argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](gen: T, n: Int): Vec[T] = new Vec(gen.cloneType, n)

  /** Creates a new [[Vec]] composed of elements of the input Seq of [[Data]]
    * nodes.
    *
    * @note input elements should be of the same type
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](elts: Seq[T]): Vec[T] = {
    // REVIEW TODO: error checking to guard against type mismatch?

    require(!elts.isEmpty)
    val width = elts.map(_.width).reduce(_ max _)
    val vec = new Vec(elts.head.cloneTypeWidth(width), elts.length)
    pushCommand(DefWire(vec))
    for ((v, e) <- vec zip elts)
      v := e
    vec
  }

  /** Creates a new [[Vec]] composed of the input [[Data]] nodes.
    *
    * @note input elements should be of the same type
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](elt0: T, elts: T*): Vec[T] =
    // REVIEW TODO: does this really need to exist as a standard function?
    apply(elt0 +: elts.toSeq)

  /** Creates a new [[Vec]] of length `n` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of elements in the vector (the function is applied from
    * 0 to `n-1`)
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] =
    apply((0 until n).map(i => gen(i)))

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function repeatedly applied.
    *
    * @param n number of elements (amd the number of times the function is
    * called)
    * @param gen function that generates the [[Data]] that becomes the output
    * element
    */
  def fill[T <: Data](n: Int)(gen: => T): Vec[T] = apply(Seq.fill(n)(gen))
}

/** An abstract class for data types that solely consist of (are an aggregate
  * of) other Data objects.
  */
sealed abstract class Aggregate(dirArg: Direction) extends Data(dirArg) {
  private[Chisel] def cloneTypeWidth(width: Width): this.type = cloneType
  def width: Width = flatten.map(_.width).reduce(_ + _)
}

/** A vector (array) of [[Data]] elements. Provides hardware versions of various
  * collection transformation functions found in software array implementations.
  *
  * @tparam T type of elements
  */
sealed class Vec[T <: Data] private (gen: => T, val length: Int)
    extends Aggregate(gen.dir) with VecLike[T] {
  // REVIEW TODO: should this take a Seq instead of a gen()?

  private val self = IndexedSeq.fill(length)(gen)

  override def <> (that: Data): Unit = that match {
    case _: Vec[_] => this bulkConnect that
    case _ => this badConnect that
  }

  def <> (that: Seq[T]): Unit =
    // REVIEW TODO: come up with common style: match on type in body or
    // multiple invocation signatures
    for ((a, b) <- this zip that)
      a <> b

  def <> (that: Vec[T]): Unit = this bulkConnect that
    // REVIEW TODO: standardize as above

  override def := (that: Data): Unit = that match {
    case _: Vec[_] => this connect that
    case _ => this badConnect that
  }

  def := (that: Seq[T]): Unit = {
    // REVIEW TODO: standardize as above
    require(this.length == that.length)
    for ((a, b) <- this zip that)
      a := b
  }

  def := (that: Vec[T]): Unit = this connect that

  /** Creates a dynamically indexed read accessor into the array. Generates
    * logic (likely some kind of multiplexer).
    */
  def apply(idx: UInt): T = {
    val x = gen
    // REVIEW TODO: what happens when people try to assign into this?
    // Should this be a read-only reference?
    pushCommand(DefAccessor(x, Node(this), NO_DIR, idx.ref))
    x
  }

  /** Creates a statically indexed read accessor into the array. Generates no
    * logic.
    */
  def apply(idx: Int): T = self(idx)

  def read(idx: UInt): T = apply(idx)
  // REVIEW TODO: does this need to exist?

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

/** A trait for [[Vec]]s containing common hardware generators for collection
  * operations.
  */
trait VecLike[T <: Data] extends collection.IndexedSeq[T] {
  def read(idx: UInt): T
  // REVIEW TODO: does this need to exist? (does the same thing as apply)

  def write(idx: UInt, data: T): Unit
  def apply(idx: UInt): T

  /** Outputs true if p outputs true for every element.
    *
    * This generates into a function evaluation followed by a logical AND
    * reduction.
    */
  def forall(p: T => Bool): Bool = (this map p).fold(Bool(true))(_ && _)

  /** Outputs true if p outputs true for at least one element.
    *
    * This generates into a function evaluation followed by a logical OR
    * reduction.
    */
  def exists(p: T => Bool): Bool = (this map p).fold(Bool(false))(_ || _)

  /** Outputs true if the vector contains at least one element equal to x (using
    * the === operator).
    *
    * This generates into an equality comparison followed by a logical OR
    * reduction.
    */
  def contains(x: T)(implicit evidence: T <:< UInt): Bool = this.exists(_ === x)

  /** Outputs the number of elements for which p is true.
    *
    * This generates into a function evaluation followed by a set bit counter.
    */
  def count(p: T => Bool): UInt = PopCount((this map p).toSeq)

  /** Helper function that appends an index (literal value) to each element,
    * useful for hardware generators which output an index.
    */
  private def indexWhereHelper(p: T => Bool) = this map p zip (0 until length).map(i => UInt(i))

  /** Outputs the index of the first element for which p outputs true.
    *
    * This generates into a function evaluation followed by a priority mux.
    */
  def indexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p))

  /** Outputs the index of the last element for which p outputs true.
    *
    * This generates into a function evaluation followed by a priority mux.
    */
  def lastIndexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p).reverse)

  /** Outputs the index of the element for which p outputs true, assuming that
    * the there is exactly one such element.
    *
    * This generates into a function evaluation followed by a one-hot mux. The
    * implementation may be more efficient than a priority mux, but incorrect
    * results are possible if there is not exactly one true element.
    */
  def onlyIndexWhere(p: T => Bool): UInt = Mux1H(indexWhereHelper(p))
  // REVIEW TODO: can (should?) this be assertion checked?
}

object BitPat {
  /** Parses a bit pattern string into (bits, mask, width).
    *
    * @return bits the literal value, with don't cares being 0
    * @return mask the mask bits, with don't cares being 0 and cares being 1
    * @return width the number of bits in the literal, including values and
    * don't cares.
    */
  private def parse(x: String): (BigInt, BigInt, Int) = {
    // REVIEW TODO: can this be merged with literal parsing creating one unified
    // Chisel string to value decoder (which can also be invoked by libraries
    // and testbenches?
    // REVIEW TODO: Verilog Xs also handle octal and hex cases.
    require(x.head == 'b', "BitPats must be in binary and be prefixed with 'b'")
    var bits = BigInt(0)
    var mask = BigInt(0)
    for (d <- x.tail) {
      if (d != '_') {
        if (!"01?".contains(d)) Builder.error({"Literal: " + x + " contains illegal character: " + d})
        mask = (mask << 1) + (if (d == '?') 0 else 1)
        bits = (bits << 1) + (if (d == '1') 1 else 0)
      }
    }
    (bits, mask, x.length - 1)
  }

  /** Creates a [[BitPat]] literal from a string.
   *
    * @param n the literal value as a string, in binary, prefixed with 'b'
    * @note legal characters are '0', '1', and '?', as well as '_' as white
    * space (which are ignored)
    */
  def apply(n: String): BitPat = {
    val (bits, mask, width) = parse(n)
    new BitPat(bits, mask, width)
  }

  /** Creates a [[BitPat]] of all don't cares of a specified width. */
  // REVIEW TODO: is this really necessary? if so, can there be a better name?
  def DC(width: Int): BitPat = BitPat("b" + ("?" * width))

  // BitPat <-> UInt
  /** enable conversion of a bit pattern to a UInt */
  // REVIEW TODO: Doesn't having a BitPat with all mask bits high defeat the
  // point of using a BitPat in the first place?
  implicit def BitPatToUInt(x: BitPat): UInt = {
    require(x.mask == (BigInt(1) << x.getWidth) - 1)
    UInt(x.value, x.getWidth)
  }

  /** create a bit pattern from a UInt */
  // REVIEW TODO: Similar, what is the point of this?
  implicit def apply(x: UInt): BitPat = {
    require(x.isLit)
    BitPat("b" + x.litValue.toString(2))
  }
}

// TODO: Break out of Core? (this doesn't involve FIRRTL generation)
/** Bit patterns are literals with masks, used to represent values with don't
  * cares. Equality comparisons will ignore don't care bits (for example,
  * BitPat(0b10?1) === UInt(0b1001) and UInt(0b1011)).
  */
sealed class BitPat(val value: BigInt, val mask: BigInt, width: Int) {
  def getWidth: Int = width
  def === (other: UInt): Bool = UInt(value) === (other & UInt(mask))
  def != (other: UInt): Bool = !(this === other)
}

/** Element is a leaf data type: it cannot contain other Data objects. Example
  * uses are for representing primitive data types, like integers and bits.
  */
abstract class Element(dirArg: Direction, val width: Width) extends Data(dirArg) {
  // REVIEW TODO: toBits is implemented in terms of flatten... inheriting this
  // without rewriting toBits will break things. Perhaps have a specific element
  // API?
  private[Chisel] def flatten: IndexedSeq[UInt] = IndexedSeq(toBits)
}

object Clock {
  def apply(dir: Direction = NO_DIR): Clock = new Clock(dir)
}

// TODO: Document this.
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

/** A data type for values represented by a single bitvector. Provides basic
  * bitwise operations.
  */
sealed abstract class Bits(dirArg: Direction, width: Width, override val litArg: Option[LitArg])
    extends Element(dirArg, width) {
  // REVIEW TODO: should this be abstract? It may be good to use Bits for values
  // where you don't need artihmetic operations / arithmetic doesn't make sense
  // like opcodes and stuff.

  // REVIEW TODO: Why do we need a fromInt? Why does it drop the argument?
  def fromInt(x: BigInt): this.type
  // REVIEW TODO: purpose of dedicated lit logic?
  def makeLit(value: BigInt): LitArg
  def cloneType: this.type = cloneTypeWidth(width)

  override def <> (that: Data): Unit = this := that

  /** Returns the specified bit on this wire as a [[Bool]], statically
    * addressed. Generates no logic.
    */
  // REVIEW TODO: Ddeduplicate constructor with apply(Int)
  final def apply(x: BigInt): Bool = {
    if (x < 0) {
      Builder.error(s"Negative bit indices are illegal (got $x)")
    }
    if (isLit()) {
      Bool(((litValue() >> x.toInt) & 1) == 1)
    } else {
      pushOp(DefPrim(Bool(), BitSelectOp, this.ref, ILit(x)))
    }
  }

  final def apply(x: Int): Bool =
    apply(BigInt(x))

  /** Returns the specified bit on this wire as a [[Bool]], dynamically
    * addressed. Generates logic: implemented as a variable shifter.
    */
  final def apply(x: UInt): Bool =
    (this >> x)(0)

  /** Returns a subset of bits on this wire from `hi` to `lo` (inclusive),
    * statically addressed. Generates no logic.
    *
    * @example
    * {{{
    * myBits = 0x5 = 0b101
    * myBits(1,0) => 0b01  // extracts the two least significant bits
    * }}}
    */
  final def apply(x: Int, y: Int): UInt = {
    if (x < y || y < 0) {
      Builder.error(s"Invalid bit range ($x,$y)")
    }
    // REVIEW TODO: should we support negative indexing Python style, at least
    // where widths are known?
    val w = x - y + 1
    if (isLit()) {
      UInt((litValue >> y) & ((BigInt(1) << w) - 1), w)
    } else {
      pushOp(DefPrim(UInt(width = w), BitsExtractOp, this.ref, ILit(x), ILit(y)))
    }
  }

  // REVIEW TODO: again, is this necessary? Or just have this and use implicits?
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

  /** Returns this wire bitwise-inverted. */
  def unary_~ : this.type = unop(cloneTypeWidth(width), BitNotOp)

  /** Returns this wire zero padded up to the specified width.
    *
    * @note for SInts only, this does sign extension
    */
  def pad (other: Int): this.type = binop(cloneTypeWidth(this.width max Width(other)), PadOp, other)

  /** Shift left operation */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or Bits?
  def << (other: BigInt): Bits

  /** Returns this wire statically left shifted by the specified amount,
    * inserting zeros into the least significant bits.
    *
    * The width of the output is `other` larger than the input. Generates no
    * logic.
    */
  def << (other: Int): Bits

  /** Returns this wire dynamically left shifted by the specified amount,
    * inserting zeros into the least significant bits.
    *
    * The width of the output is `pow(2, width(other))` larger than the input.
    * Generates a dynamic shifter.
    */
  def << (other: UInt): Bits

  /** Shift right operation */
  // REVIEW TODO: redundant
  def >> (other: BigInt): Bits

  /** Returns this wire statically right shifted by the specified amount,
    * inserting zeros into the most significant bits.
    *
    * The width of the output is the same as the input. Generates no logic.
    */
  def >> (other: Int): Bits

  /** Returns this wire dynamically right shifted by the specified amount,
    * inserting zeros into the most significant bits.
    *
    * The width of the output is the same as the input. Generates a dynamic
    * shifter.
    */
  def >> (other: UInt): Bits

  /** Returns the contents of this wire as a [[Vec]] of [[Bool]]s. Generates no
    * logic.
    */
  def toBools: Vec[Bool] = Vec.tabulate(this.getWidth)(i => this(i))

  // REVIEW TODO: is this appropriate here? Should this be a (implicit?) cast in
  // the SInt object instead? Bits shouldn't know about UInt/SInt, which are
  // downstream?
  def asSInt(): SInt
  def asUInt(): UInt
  final def toSInt(): SInt = asSInt
  final def toUInt(): UInt = asUInt

  def toBool(): Bool = width match {
    case KnownWidth(1) => this(0)
    case _ => throwException(s"can't covert UInt<$width> to Bool")
  }

  // REVIEW TODO: where did this syntax come from?
  /** Returns this wire concatenated with `other`, where this wire forms the
    * most significant part and `other` forms the least significant part.
    *
    * The width of the output is sum of the inputs. Generates no logic.
    */
  def ## (other: Bits): UInt = Cat(this, other)

  // REVIEW TODO: This just _looks_ wrong.
  override def toBits: UInt = asUInt

  override def fromBits(n: Bits): this.type = {
    val res = Wire(this).asInstanceOf[this.type]
    res := n
    res
  }
}

// REVIEW TODO: Numeric (strictly UInt/SInt/float) or numeric-like (complex,
// etc)? First is easy to define, second not so much. Perhaps rename IntLike?
// Also should add intended purpose.
/** Abstract trait defining operations available on numeric-like wire data
  * types.
  */
abstract trait Num[T <: Data] {
  // def << (b: T): T;
  // def >> (b: T): T;
  //def unary_-(): T;

  // REVIEW TODO: double check ops conventions against FIRRTL

  /** Outputs the sum of `this` and `b`. The resulting width is the max of the
    * operands plus 1 (should not overflow).
    */
  def +  (b: T): T;

  /** Outputs the product of `this` and `b`. The resulting width is the sum of
    * the operands.
    *
    * @note can generate a single-cycle multiplier, which can result in
    * significant cycle time and area costs
    */
  def *  (b: T): T;

  /** Outputs the quotient of `this` and `b`.
    *
    * TODO: full rules
    */
  def /  (b: T): T;

  def %  (b: T): T;

  /** Outputs the difference of `this` and `b`. The resulting width is the max
   *  of the operands plus 1 (should not overflow).
    */
  def -  (b: T): T;

  /** Outputs true if `this` < `b`.
    */
  def <  (b: T): Bool;

  /** Outputs true if `this` <= `b`.
    */
  def <= (b: T): Bool;

  /** Outputs true if `this` > `b`.
    */
  def >  (b: T): Bool;

  /** Outputs true if `this` >= `b`.
    */
  def >= (b: T): Bool;

  /** Outputs the minimum of `this` and `b`. The resulting width is the max of
    * the operands. Generates a comparison followed by a mux.
    */
  def min(b: T): T = Mux(this < b, this.asInstanceOf[T], b)

  /** Outputs the maximum of `this` and `b`. The resulting width is the max of
    * the operands. Generates a comparison followed by a mux.
    */
  def max(b: T): T = Mux(this < b, b, this.asInstanceOf[T])
}

/** A data type for unsigned integers, represented as a binary bitvector.
  * Defines arithmetic operations between other integer types.
  */
sealed class UInt private[Chisel] (dir: Direction, width: Width, lit: Option[ULit] = None)
    extends Bits(dir, width, lit) with Num[UInt] {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type =
    new UInt(dir, w).asInstanceOf[this.type]
  private[Chisel] def toType = s"UInt<$width>"

  def fromInt(value: BigInt): this.type = UInt(value).asInstanceOf[this.type]
  def makeLit(value: BigInt): ULit = ULit(value, Width())

  override def := (that: Data): Unit = that match {
    case _: UInt => this connect that
    case _ => this badConnect that
  }

  // TODO: refactor to share documentation with Num or add independent scaladoc
  def unary_- : UInt = UInt(0) - this
  def unary_-% : UInt = UInt(0) -% this
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

  // REVIEW TODO: Can this be defined on Bits?
  def orR: Bool = this != UInt(0)
  def andR: Bool = ~this === UInt(0)
  def xorR: Bool = redop(XorReduceOp)

  def < (other: UInt): Bool = compop(LessOp, other)
  def > (other: UInt): Bool = compop(GreaterOp, other)
  def <= (other: UInt): Bool = compop(LessEqOp, other)
  def >= (other: UInt): Bool = compop(GreaterEqOp, other)
  def != (other: UInt): Bool = compop(NotEqualOp, other)
  def === (other: UInt): Bool = compop(EqualOp, other)
  def unary_! : Bool = this === Bits(0)

  // REVIEW TODO: Can these also not be defined on Bits?
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

  // REVIEW TODO: Is this really the common definition of zero extend?
  // Can we just define UInt/SInt constructors on Bits as a reinterpret case?
  /** Returns this UInt as a [[SInt]] with an additional zero in the MSB.
    */
  def zext(): SInt = pushOp(DefPrim(SInt(width + 1), ConvertOp, ref))

  /** Returns this UInt as a [[SInt]], without changing width or bit value. The
    * SInt is not guaranteed to have the same value (for example, if the MSB is
    * high, it will be interpreted as a negative value).
    */
  def asSInt(): SInt = pushOp(DefPrim(SInt(width), AsSIntOp, ref))

  def asUInt(): UInt = this
}

// REVIEW TODO: why not just have this be a companion object? Why the trait
// instead of object UInt?
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

  private def parse(n: String) = {
    val (base, num) = n.splitAt(1)
    val radix = base match {
      case "x" | "h" => 16
      case "d" => 10
      case "o" => 8
      case "b" => 2
      case _ => Builder.error(s"Invalid base $base"); 2
    }
    BigInt(num, radix)
  }

  private def parsedWidth(n: String) =
    if (n(0) == 'b') {
      Width(n.length-1)
    } else if (n(0) == 'h') {
      Width((n.length-1) * 4)
    } else {
      Width()
    }
}

object UInt extends UIntFactory

// REVIEW TODO: Wait, wha?! Why does this exist? Things should be DRY and
// unambiguous.
/** Provides a set of operations to create UInt types and literals.
  * Identical in functionality to the UInt companion object. */
object Bits extends UIntFactory

sealed class SInt private (dir: Direction, width: Width, lit: Option[SLit] = None)
    extends Bits(dir, width, lit) with Num[SInt] {
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

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  */
sealed class Bool(dir: Direction, lit: Option[ULit] = None) extends UInt(dir, Width(1), lit) {
  private[Chisel] override def cloneTypeWidth(w: Width): this.type = {
    require(!w.known || w.get == 1)
    new Bool(dir).asInstanceOf[this.type]
  }

  override def fromInt(value: BigInt): this.type = {
    require(value == 0 || value == 1)
    Bool(value == 1).asInstanceOf[this.type]
  }

  // REVIEW TODO: Why does this need to exist and have different conventions
  // than Bits?
  def & (other: Bool): Bool = binop(Bool(), BitAndOp, other)
  def | (other: Bool): Bool = binop(Bool(), BitOrOp, other)
  def ^ (other: Bool): Bool = binop(Bool(), BitXorOp, other)

  /** Outputs the logical OR of two Bools.
   */
  def || (that: Bool): Bool = this | that

  /** Outputs the logical AND of two Bools.
   */
  def && (that: Bool): Bool = this & that
}

object Bool {
  /** Creates an empty Bool.
   */
  def apply(dir: Direction = NO_DIR): Bool = new Bool(dir)

  /** Creates Bool literal.
   */
  def apply(x: Boolean): Bool = new Bool(NO_DIR, Some(ULit(if (x) 1 else 0, Width(1))))
}

object Mux {
  /** Creates a mux, whose output is one of the inputs depending on the
    * value of the condition.
    *
    * @param cond condition determining the input to choose
    * @param con the value chosen when `cond` is true
    * @param alt the value chosen when `cond` is false
    * @example
    * {{{
    * val muxOut = Mux(data_in === UInt(3), UInt(3, 4), UInt(0, 4))
    * }}}
    */
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

// REVIEW TODO: Should the FIRRTL emission be part of Bits, with a separate
// Cat in stdlib that can do a reduction among multiple elements?
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
    if (r.tail.isEmpty) {
      r.head.asUInt
    } else {
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

/** Base class for data types defined as a bundle of other data types.
  *
  * Usage: extend this class (either as an anonymous or named class) and define
  * members variables of [[Data]] subtypes to be elements in the Bundle.
  */
class Bundle extends Aggregate(NO_DIR) {
  private val _namespace = Builder.globalNamespace.child

  // REVIEW TODO: perhaps deprecate to match FIRRTL semantics? Also needs
  // strong connect operator.
  /** Connect elements in this Bundle to elements in `that` on a best-effort
    * (weak) basis, matching by type, orientation, and name.
    *
    * @note unconnected elements will NOT generate errors or warnings
    *
    * @example
    * {{{
    * // Pass through wires in this module's io to those mySubModule's io,
    * // matching by type, orientation, and name, and ignoring extra wires.
    * mySubModule.io <> io
    * }}}
    */
  override def <> (that: Data): Unit = that match {
    case _: Bundle => this bulkConnect that
    case _ => this badConnect that
  }

  // REVIEW TODO: should there be different semantics for this? Or just ban it?
  override def := (that: Data): Unit = this <> that

  lazy val elements: ListMap[String, Data] = ListMap(namedElts:_*)

  /** Returns a best guess at whether a field in this Bundle is a user-defined
    * Bundle element.
    */
  private def isBundleField(m: java.lang.reflect.Method) =
    m.getParameterTypes.isEmpty &&
    !java.lang.reflect.Modifier.isStatic(m.getModifiers) &&
    classOf[Data].isAssignableFrom(m.getReturnType) &&
    !(Bundle.keywords contains m.getName) && !(m.getName contains '$')

  /** Returns a list of elements in this Bundle.
    */
  private[Chisel] lazy val namedElts = {
    val nameMap = LinkedHashMap[String, Data]()
    val seen = HashSet[Data]()
    for (m <- getClass.getMethods.sortWith(_.getName < _.getName); if isBundleField(m)) {
      m.invoke(this) match {
        case d: Data =>
          if (nameMap contains m.getName) {
            require(nameMap(m.getName) eq d)
          } else if (!seen(d)) {
            nameMap(m.getName) = d; seen += d
          }
        case _ =>
      }
    }
    ArrayBuffer(nameMap.toSeq:_*) sortWith {case ((an, a), (bn, b)) => (a._id > b._id) || ((a eq b) && (an > bn))}
  }
  private[Chisel] def toType = {
    def eltPort(elt: Data): String = {
      val flipStr = if (elt.isFlip) "flip " else ""
      s"${flipStr}${elt.getRef.name} : ${elt.toType}"
    }
    s"{${namedElts.reverse.map(e => eltPort(e._2)).mkString(", ")}}"
  }
  private[Chisel] lazy val flatten = namedElts.flatMap(_._2.flatten)
  private[Chisel] def addElt(name: String, elt: Data): Unit =
    namedElts += name -> elt
  private[Chisel] override def _onModuleClose: Unit =
    for ((name, elt) <- namedElts) { elt.setRef(this, _namespace.name(name)) }

  override def cloneType : this.type = {
    // If the user did not provide a cloneType method, try invoking one of
    // the following constructors, not all of which necessarily exist:
    // - A zero-parameter constructor
    // - A one-paramater constructor, with null as the argument
    // - A one-parameter constructor for a nested Bundle, with the enclosing
    //   parent Module as the argument
    val constructor = this.getClass.getConstructors.head
    try {
      val args = Seq.fill(constructor.getParameterTypes.size)(null)
      constructor.newInstance(args:_*).asInstanceOf[this.type]
    } catch {
      case e: java.lang.reflect.InvocationTargetException if e.getCause.isInstanceOf[java.lang.NullPointerException] =>
        try {
          constructor.newInstance(_parent.get).asInstanceOf[this.type]
        } catch {
          case _: java.lang.reflect.InvocationTargetException =>
            Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method. You are probably using " +
              "an anonymous Bundle object that captures external state and hence is un-cloneTypeable")
            this
        }
      case _: java.lang.reflect.InvocationTargetException | _: java.lang.IllegalArgumentException =>
        Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method")
        this
    }
  }
}

object Module {
  // TODO: update documentation when parameters gets removed from core Chisel
  // and this gets simplified.
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param m the Module being created
    * @param p Parameters passed down implicitly from that it is created in
    *
    * @return the input module `m`
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

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class Module(_clock: Clock = null, _reset: Bool = null) extends HasId {
  private val _namespace = Builder.globalNamespace.child
  private[Chisel] val _commands = ArrayBuffer[Command]()
  private[Chisel] val _ids = ArrayBuffer[HasId]()
  dynamicContext.currentModule = Some(this)

  /** Name of the instance. */
  val name = Builder.globalNamespace.name(getClass.getName.split('.').last)

  /** IO for this Module. At the Scala level (pre-FIRRTL transformations),
    * connections in and out of a Module may only go through `io` elements.
    */
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

/** Defines a black box, which is a module that can be referenced from within
  * Chisel, but is not defined in the emitted Verilog. Useful for connecting
  * to RTL modules defined outside Chisel.
  *
  * @example
  * {{{
  * class DSP48E1 extends BlackBox {
  *   val io = new Bundle // Create I/O with same as DSP
  *   val dspParams = new VerilogParameters // Create Parameters to be specified
  *   setVerilogParams(dspParams)
  *   // Implement functionality of DSP to allow simulation verification
  * }
  * }}}
  */
// TODO: actually implement BlackBox (this hack just allows them to compile)
// REVIEW TODO: make Verilog parameters part of the constructor interface?
abstract class BlackBox(_clock: Clock = null, _reset: Bool = null) extends Module(_clock = _clock, _reset = _reset) {
  def setVerilogParameters(s: String): Unit = {}
}

object when {
  /** Create a `when` condition block, where whether a block of logic is
    * executed or not depends on the conditional.
    *
    * @param cond condition to execute upon
    * @param block logic that runs only if `cond` is true
    *
    * @example
    * {{{
    * when ( myData === UInt(3) ) {
    *   // Some logic to run when myData equals 3.
    * } .elsewhen ( myData === UInt(1) ) {
    *   // Some logic to run when myData equals 1.
    * } .otherwise {
    *   // Some logic to run when myData is neither 3 nor 1.
    * }
    * }}}
    */
  def apply(cond: => Bool)(block: => Unit): WhenContext = {
    new WhenContext(cond)(block)
  }
}

class WhenContext(cond: => Bool)(block: => Unit) {
  /** This block of logic gets executed if above conditions have been false
    * and this condition is true.
    */
  def elsewhen (cond: => Bool)(block: => Unit): WhenContext =
    doOtherwise(when(cond)(block))

  /** This block of logic gets executed only if the above conditions were all
    * false. No additional logic blocks may be appended past the `otherwise`.
    */
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

// TODO: check with FIRRTL specs, how much official implementation flexibility
// is there?
/** A source of garbage data, used to initialize Wires to a don't-care value. */
private object Poison extends Command {
  def apply[T <: Data](t: T): T =
    pushCommand(DefPoison(t.cloneType)).id
}
