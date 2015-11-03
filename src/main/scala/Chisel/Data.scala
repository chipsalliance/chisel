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

@deprecated("debug doesn't do anything in Chisel3 as no pruning happens in the frontend", "chisel3")
object debug {  // scalastyle:ignore object.name
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

  def := (that: Data): Unit = this badConnect that
  def <> (that: Data): Unit = this badConnect that
  def cloneType: this.type
  def litArg(): Option[LitArg] = None
  def litValue(): BigInt = litArg.get.num
  def isLit(): Boolean = litArg.isDefined

  def width: Width
  final def getWidth: Int = width.get

  // While this being in the Data API doesn't really make sense (should be in
  // Aggregate, right?) this is because of an implementation limitation:
  // cloneWithDirection, which is private and defined here, needs flatten to
  // set element directionality.
  // Related: directionality is mutable state. A possible solution for both is
  // to define directionality relative to the container, but these parent links
  // currently don't exist (while this information may be available during
  // FIRRTL emission, it would break directionality querying from Chisel, which
  // does get used).
  private[Chisel] def flatten: IndexedSeq[Bits]

  /** Creates an new instance of this type, unpacking the input Bits into
    * structured data.
    *
    * This performs the inverse operation of toBits.
    *
    * @note does NOT assign to the object this is called on, instead creates
    * and returns a NEW object (useful in a clone-and-assign scenario)
    * @note does NOT check bit widths, may drop bits during assignment
    * @note what fromBits assigs to must have known widths
    */
  def fromBits(n: Bits): this.type = {
    var i = 0
    val wire = Wire(this.cloneType)
    for (x <- wire.flatten) {
      x := n(i + x.getWidth-1, i)
      i += x.getWidth
    }
    wire.asInstanceOf[this.type]
  }

  /** Packs the value of this object as plain Bits.
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
