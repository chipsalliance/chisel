// See LICENSE for license details.

package Chisel

import internal._
import internal.Builder.pushCommand
import internal.firrtl._

sealed abstract class Direction(name: String) {
  override def toString: String = name
  def flip: Direction
}
object Direction {
  object Input  extends Direction("input") { override def flip: Direction = Output }
  object Output extends Direction("output") { override def flip: Direction = Input }
}

@deprecated("debug doesn't do anything in Chisel3 as no pruning happens in the frontend", "chisel3")
object debug {  // scalastyle:ignore object.name
  def apply (arg: Data): Data = arg
}

object DataMirror {
  def widthOf(target: Data): Width = target.width
}

/**
* Input, Output, and Flipped are used to define the directions of Module IOs.
*
* Note that they do not currently call target to be a newType or cloneType.
* This is nominally for performance reasons to avoid too many extra copies when
* something is flipped multiple times.
*
* Thus, an error will be thrown if these are used on bound Data
*/
object Input {
  def apply[T<:Data](target: T): T =
    Binding.bind(target, InputBinder, "Error: Cannot set as input ")
}
object Output {
  def apply[T<:Data](target: T): T =
    Binding.bind(target, OutputBinder, "Error: Cannot set as output ")
}
object Flipped {
  def apply[T<:Data](target: T): T =
    Binding.bind(target, FlippedBinder, "Error: Cannot flip ")
}

object Data {
  /**
  * This function returns true if the FIRRTL type of this Data should be flipped
  * relative to other nodes.
  *
  * Note that the current scheme only applies Flip to Elements or Vec chains of
  * Elements.
  *
  * A Bundle is never marked flip, instead preferring its root fields to be marked
  *
  * The Vec check is due to the fact that flip must be factored out of the vec, ie:
  * must have flip field: Vec(UInt) instead of field: Vec(flip UInt)
  */
  private[Chisel] def isFlipped(target: Data): Boolean = target match {
    case (element: Element) => element.binding.direction == Some(Direction.Input)
    case (vec: Vec[Data @unchecked]) => isFlipped(vec.sample_element)
    case (bundle: Bundle) => false
  }

  implicit class AddDirectionToData[T<:Data](val target: T) extends AnyVal {
    @deprecated("Input(Data) should be used over Data.asInput", "gchisel")
    def asInput: T = Input(target)
    @deprecated("Output(Data) should be used over Data.asOutput", "gchisel")
    def asOutput: T = Output(target)
    @deprecated("Flipped(Data) should be used over Data.flip", "gchisel")
    def asFlip: T = Flipped(target)
  }
}

/** This forms the root of the type system for wire data types. The data value
  * must be representable as some number (need not be known at Chisel compile
  * time) of bits, and must have methods to pack / unpack structured data to /
  * from bits.
  */
abstract class Data extends HasId {
  // Return ALL elements at root of this type.
  // Contasts with flatten, which returns just Bits
  private[Chisel] def allElements: Seq[Element]

  private[Chisel] def badConnect(that: Data): Unit =
    throwException(s"cannot connect ${this} and ${that}")
  private[Chisel] def connect(that: Data): Unit = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(that, s"'that' ($that)")
    try {
      MonoConnect.connect(this, that, Builder.forcedModule)
    } catch {
      case MonoConnect.MonoConnectException(message) =>
        throw new Exception(
          s"Connection between sink ($this) and source ($that) failed @$message"
        )
    }
  }
  private[Chisel] def bulkConnect(that: Data): Unit = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(that, s"'that' ($that)")
    try {
      BiConnect.connect(this, that, Builder.forcedModule)
    } catch {
      case BiConnect.BiConnectException(message) =>
        throw new Exception(
          s"Connection between left ($this) and source ($that) failed @$message"
        )
    }
  }
  private[Chisel] def lref: Node = Node(this)
  private[Chisel] def ref: Arg = if (isLit) litArg.get else lref
  private[Chisel] def toType: String

  final def := (that: Data): Unit = this connect that
  final def <> (that: Data): Unit = this bulkConnect that
  def litArg(): Option[LitArg] = None
  def litValue(): BigInt = litArg.get.num
  def isLit(): Boolean = litArg.isDefined

  private[Chisel] def width: Width
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
    val bits =
      if (n.width.known && n.width.get >= wire.width.get) n
      else Wire(n.cloneTypeWidth(wire.width), init = n)
    for (x <- wire.flatten) {
      x := bits(i + x.getWidth-1, i)
      i += x.getWidth
    }
    wire.asInstanceOf[this.type]
  }

  /** Packs the value of this object as plain Bits.
    *
    * This performs the inverse operation of fromBits(Bits).
    */
  def toBits(): UInt = SeqUtils.asUInt(this.flatten)

  //TODO(twigg): Remove cloneTypeWidth, it doesn't compose for aggregates....
  private[Chisel] def cloneTypeWidth(width: Width): this.type

  protected def cloneType: this.type
  final def newType: this.type = {
    val clone = this.cloneType
    //TODO(twigg): Do recursively for better error messages
    for((clone_elem, source_elem) <- clone.allElements zip this.allElements) {
      clone_elem.binding = UnboundBinding(source_elem.binding.direction)
    }
    clone
  }
}

object Wire {
  def apply[T <: Data](t: T): T =
    makeWire(t, null.asInstanceOf[T])

  def apply[T <: Data](dummy: Int = 0, init: T): T =
    makeWire(null.asInstanceOf[T], init)

  def apply[T <: Data](t: T, init: T): T =
    makeWire(t, init)

  private def makeWire[T <: Data](t: T, init: T): T = {
    val x = Reg.makeType(t, null.asInstanceOf[T], init)

    // Bind each element of x to being a Wire
    Binding.bind(x, WireBinder(Builder.forcedModule), "Error: t")

    pushCommand(DefWire(x))
    pushCommand(DefInvalid(x.ref))
    if (init != null) {
      Binding.checkSynthesizable(init, s"'init' ($init)")
      x := init
    }

    x
  }
}

object Clock {
  def apply(): Clock = new Clock
}

// TODO: Document this.
sealed class Clock extends Element(Width(1)) {
  def cloneType: this.type = Clock().asInstanceOf[this.type]
  private[Chisel] override def flatten: IndexedSeq[Bits] = IndexedSeq()
  private[Chisel] def cloneTypeWidth(width: Width): this.type = cloneType
  private[Chisel] def toType = "Clock"
}
