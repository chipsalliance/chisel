// See LICENSE for license details.

package Chisel
import Builder.pushOp
import PrimOp._

/** Element is a leaf data type: it cannot contain other Data objects. Example
  * uses are for representing primitive data types, like integers and bits.
  */
abstract class Element(dirArg: Direction, val width: Width) extends Data(dirArg) {
  // REVIEW TODO: toBits is implemented in terms of flatten... inheriting this
  // without rewriting toBits will break things. Perhaps have a specific element
  // API?
  private[Chisel] def flatten: IndexedSeq[UInt] = IndexedSeq(toBits)
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

// REVIEW TODO: Wait, wha?! Why does this exist? Things should be DRY and
// unambiguous.
/** Provides a set of operations to create UInt types and literals.
  * Identical in functionality to the UInt companion object. */
object Bits extends UIntFactory

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

