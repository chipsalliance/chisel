// See LICENSE for license details.

package Chisel
import Builder.pushCommand

sealed abstract class Direction(name: String) {
  override def toString: String = name
  def flip: Direction
}
object INPUT  extends Direction("input") { override def flip: Direction = OUTPUT }
object OUTPUT extends Direction("output") { override def flip: Direction = INPUT }
object NO_DIR extends Direction("?") { override def flip: Direction = NO_DIR }

// REVIEW TODO: Should this actually be part of the RTL API? RTL should be
// considered untouchable from a debugging standpoint?
object debug {  // scalastyle:ignore object.name
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

// TODO: check with FIRRTL specs, how much official implementation flexibility
// is there?
/** A source of garbage data, used to initialize Wires to a don't-care value. */
private object Poison extends Command {
  def apply[T <: Data](t: T): T =
    pushCommand(DefPoison(t.cloneType)).id
}
