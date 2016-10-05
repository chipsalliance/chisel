// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, DeprecatedSourceInfo, UnlocatableSourceInfo, WireTransform, SourceInfoTransform}
import chisel3.internal.firrtl.PrimOp.AsUIntOp

sealed abstract class Direction(name: String) {
  override def toString: String = name
  def flip: Direction
}
object Direction {
  object Input  extends Direction("input") { override def flip: Direction = Output }
  object Output extends Direction("output") { override def flip: Direction = Input }
  object Unspecified extends Direction("unspecified") { override def flip: Direction = Input }
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
* Note that they currently clone their source argument, including its bindings.
*
* Thus, an error will be thrown if these are used on bound Data
*/
object Input {
  def apply[T<:Data](source: T): T = {
    val target = source.chiselCloneType
    Data.setFirrtlDirection(target, Direction.Input)
    Binding.bind(target, InputBinder, "Error: Cannot set as input ")
  }
}
object Output {
  def apply[T<:Data](source: T): T = {
    val target = source.chiselCloneType
    Data.setFirrtlDirection(target, Direction.Output)
    Binding.bind(target, OutputBinder, "Error: Cannot set as output ")
  }
}
object Flipped {
  def apply[T<:Data](source: T): T = {
    val target = source.chiselCloneType
    Data.setFirrtlDirection(target, Data.getFirrtlDirection(source).flip)
    Binding.bind(target, FlippedBinder, "Error: Cannot flip ")
  }
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
  private[chisel3] def isFlipped(target: Data): Boolean = target match {
    case (element: Element) => element.binding.direction == Some(Direction.Input)
    case (vec: Vec[Data @unchecked]) => isFlipped(vec.sample_element)
    case (bundle: Bundle) => false
  }

  /** This function returns the "firrtl" flipped-ness for the specified object.
    *
    * @param target the object for which we want the "firrtl" flipped-ness.
    */
  private[chisel3] def isFirrtlFlipped(target: Data): Boolean = {
    Data.getFirrtlDirection(target) == Direction.Input
  }

  /** This function gets the "firrtl" direction for the specified object.
    *
    * @param target the object for which we want to get the "firrtl" direction.
    */
  private[chisel3] def getFirrtlDirection(target: Data): Direction = target match {
    case (vec: Vec[Data @unchecked]) => vec.sample_element.firrtlDirection
    case _ => target.firrtlDirection
  }

  /** This function sets the "firrtl" direction for the specified object.
    *
    * @param target the object for which we want to set the "firrtl" direction.
    */
  private[chisel3] def setFirrtlDirection(target: Data, direction: Direction): Unit = target match {
    case (vec: Vec[Data @unchecked]) => vec.sample_element.firrtlDirection = direction
    case _ => target.firrtlDirection = direction
  }

  implicit class AddDirectionToData[T<:Data](val target: T) extends AnyVal {
    def asInput(implicit opts: CompileOptions): T = {
      if (opts.deprecateOldDirectionMethods)
        Builder.deprecated("Input(Data) should be used over Data.asInput")
      Input(target)
    }
    def asOutput(implicit opts: CompileOptions): T = {
      if (opts.deprecateOldDirectionMethods)
        Builder.deprecated("Output(Data) should be used over Data.asOutput")
      Output(target)
    }
    def flip()(implicit opts: CompileOptions): T = {
      if (opts.deprecateOldDirectionMethods)
        Builder.deprecated("Flipped(Data) should be used over Data.flip")
      Flipped(target)
    }
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
  private[chisel3] def allElements: Seq[Element]

  private[core] def badConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    throwException(s"cannot connect ${this} and ${that}")
  private[chisel3] def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(that, s"'that' ($that)")
    try {
      MonoConnect.connect(sourceInfo, connectCompileOptions, this, that, Builder.forcedModule)
    } catch {
      case MonoConnect.MonoConnectException(message) =>
        throwException(
          s"Connection between sink ($this) and source ($that) failed @$message"
        )
    }
  }
  private[chisel3] def bulkConnect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = {
    Binding.checkSynthesizable(this, s"'this' ($this)")
    Binding.checkSynthesizable(that, s"'that' ($that)")
    try {
      BiConnect.connect(sourceInfo, connectCompileOptions, this, that, Builder.forcedModule)
    } catch {
      case BiConnect.BiConnectException(message) =>
        throwException(
          s"Connection between left ($this) and source ($that) failed @$message"
        )
    }
  }
  private[chisel3] def lref: Node = Node(this)
  private[chisel3] def ref: Arg = if (isLit) litArg.get else lref
  private[core] def cloneTypeWidth(width: Width): this.type
  private[chisel3] def toType: String
  private[core] def width: Width

  def cloneType: this.type
  def chiselCloneType: this.type = {
    // Call the user-supplied cloneType method
    val clone = this.cloneType
    Data.setFirrtlDirection(clone, Data.getFirrtlDirection(this))
    //TODO(twigg): Do recursively for better error messages
    for((clone_elem, source_elem) <- clone.allElements zip this.allElements) {
      clone_elem.binding = UnboundBinding(source_elem.binding.direction)
    }
    clone
  }
  final def := (that: Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit = this.connect(that)(sourceInfo, connectionCompileOptions)
  final def <> (that: Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit = this.bulkConnect(that)(sourceInfo, connectionCompileOptions)
  def litArg(): Option[LitArg] = None
  def litValue(): BigInt = litArg.get.num
  def isLit(): Boolean = litArg.isDefined

  /** Returns the width, in bits, if currently known.
    * @throws java.util.NoSuchElementException if the width is not known. */
  final def getWidth: Int = width.get
  /** Returns whether the width is currently known. */
  final def isWidthKnown: Boolean = width.known
  /** Returns Some(width) if the width is known, else None. */
  final def widthOption: Option[Int] = if (isWidthKnown) Some(getWidth) else None

  // While this being in the Data API doesn't really make sense (should be in
  // Aggregate, right?) this is because of an implementation limitation:
  // cloneWithDirection, which is private and defined here, needs flatten to
  // set element directionality.
  // Related: directionality is mutable state. A possible solution for both is
  // to define directionality relative to the container, but these parent links
  // currently don't exist (while this information may be available during
  // FIRRTL emission, it would break directionality querying from Chisel, which
  // does get used).
  private[chisel3] def flatten: IndexedSeq[Bits]

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
  def fromBits(that: Bits): this.type = macro SourceInfoTransform.thatArg

  def do_fromBits(that: Bits)(implicit sourceInfo: SourceInfo): this.type = {
    var i = 0
    val wire = Wire(this.chiselCloneType)
    val bits =
      if (that.width.known && that.width.get >= wire.width.get) {
        that
      } else {
        Wire(that.cloneTypeWidth(wire.width), init = that)
      }
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
  @deprecated("Use asUInt, which does the same thing but makes the reinterpret cast more explicit", "chisel3")
  def toBits(): UInt = SeqUtils.do_asUInt(this.flatten)(DeprecatedSourceInfo)

  /** Reinterpret cast to UInt.
    *
    * @note value not guaranteed to be preserved: for example, a SInt of width
    * 3 and value -1 (0b111) would become an UInt with value 7
    * @note Aggregates are recursively packed with the first element appearing
    * in the least-significant bits of the result.
    */
  final def asUInt(): UInt = macro SourceInfoTransform.noArg

  def do_asUInt(implicit sourceInfo: SourceInfo): UInt =
    SeqUtils.do_asUInt(this.flatten)(sourceInfo)

  // firrtlDirection is the direction we report to firrtl.
  // It maintains the user-specified value (as opposed to the "actual" or applied/propagated value).
  // NOTE: This should only be used for emitting acceptable firrtl.
  // The Element.dir should be used for any tests involving direction.
  private var firrtlDirection: Direction = Direction.Unspecified
  /** Default pretty printing */
  def toPrintable: Printable
}

object Wire {
  def apply[T <: Data](t: T): T = macro WireTransform.apply[T]

  // No source info since Scala macros don't yet support named / default arguments.
  def apply[T <: Data](dummy: Int = 0, init: T): T =
    do_apply(null.asInstanceOf[T], init)(UnlocatableSourceInfo)

  // No source info since Scala macros don't yet support named / default arguments.
  def apply[T <: Data](t: T, init: T): T =
    do_apply(t, init)(UnlocatableSourceInfo)

  def do_apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo): T = {
    val x = Reg.makeType(chisel3.core.ExplicitCompileOptions.NotStrict, t, null.asInstanceOf[T], init)

    // Bind each element of x to being a Wire
    Binding.bind(x, WireBinder(Builder.forcedModule), "Error: t")

    pushCommand(DefWire(sourceInfo, x))
    pushCommand(DefInvalid(sourceInfo, x.ref))
    if (init != null) {
      Binding.checkSynthesizable(init, s"'init' ($init)")
      x := init
    }
    x
  }
}

object Clock {
  def apply(): Clock = new Clock
  def apply(dir: Direction): Clock = {
    val result = apply()
    dir match {
      case Direction.Input => Input(result)
      case Direction.Output => Output(result)
      case Direction.Unspecified => result
    }
  }
}

// TODO: Document this.
sealed class Clock extends Element(Width(1)) {
  def cloneType: this.type = Clock().asInstanceOf[this.type]
  private[chisel3] override def flatten: IndexedSeq[Bits] = IndexedSeq()
  private[core] def cloneTypeWidth(width: Width): this.type = cloneType
  private[chisel3] def toType = "Clock"

  override def connect (that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: Clock => super.connect(that)(sourceInfo, connectCompileOptions)
    case _ => super.badConnect(that)(sourceInfo)
  }

  /** Not really supported */
  def toPrintable: Printable = PString("CLOCK")

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
}
