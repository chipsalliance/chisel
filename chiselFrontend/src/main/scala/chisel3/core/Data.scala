// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._
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

/** Creates a clone of the super-type of the input elements. Super-type is defined as:
  * - for Bits type of the same class: the cloned type of the largest width
  * - Bools are treated as UInts
  * - For other types of the same class are are the same: clone of any of the elements
  * - Otherwise: fail
  */
private[core] object cloneSupertype {
  def apply[T <: Data](elts: Seq[T], createdType: String)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): T = {
    require(!elts.isEmpty, s"can't create $createdType with no inputs")

    if (elts forall {_.isInstanceOf[Bits]}) {
      val model: T = elts reduce { (elt1: T, elt2: T) => ((elt1, elt2) match {
        case (elt1: Bool, elt2: Bool) => elt1
        case (elt1: Bool, elt2: UInt) => elt2  // TODO: what happens with zero width UInts?
        case (elt1: UInt, elt2: Bool) => elt1  // TODO: what happens with zero width UInts?
        case (elt1: UInt, elt2: UInt) => if (elt1.width == (elt1.width max elt2.width)) elt1 else elt2  // TODO: perhaps redefine Widths to allow >= op?
        case (elt1: SInt, elt2: SInt) => if (elt1.width == (elt1.width max elt2.width)) elt1 else elt2
        case (elt1: FixedPoint, elt2: FixedPoint) => {
          require(elt1.binaryPoint == elt2.binaryPoint, s"can't create $createdType with FixedPoint with differing binaryPoints")
          if (elt1.width == (elt1.width max elt2.width)) elt1 else elt2
        }
        case (elt1, elt2) =>
          throw new AssertionError(s"can't create $createdType with heterogeneous Bits types ${elt1.getClass} and ${elt2.getClass}")
      }).asInstanceOf[T] }
      model.chiselCloneType
    } else {
      for (elt <- elts.tail) {
        require(elt.getClass == elts.head.getClass, s"can't create $createdType with heterogeneous types ${elts.head.getClass} and ${elt.getClass}")
        require(elt typeEquivalent elts.head, s"can't create $createdType with non-equivalent types ${elts.head} and ${elt}")
      }
      elts.head.chiselCloneType
    }
  }
}

/**
* Input, Output, and Flipped are used to define the directions of Module IOs.
*
* Note that they currently clone their source argument, including its bindings.
*
* Thus, an error will be thrown if these are used on bound Data
*/
object Input {
  def apply[T<:Data](source: T)(implicit compileOptions: CompileOptions): T = {
    val target = source.chiselCloneType
    Data.setFirrtlDirection(target, Direction.Input)
    Binding.bind(target, InputBinder, "Error: Cannot set as input ")
  }
}
object Output {
  def apply[T<:Data](source: T)(implicit compileOptions: CompileOptions): T = {
    val target = source.chiselCloneType
    Data.setFirrtlDirection(target, Direction.Output)
    Binding.bind(target, OutputBinder, "Error: Cannot set as output ")
  }
}
object Flipped {
  def apply[T<:Data](source: T)(implicit compileOptions: CompileOptions): T = {
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
  * A Record is never marked flip, instead preferring its root fields to be marked
  *
  * The Vec check is due to the fact that flip must be factored out of the vec, ie:
  * must have flip field: Vec(UInt) instead of field: Vec(flip UInt)
  */
  private[chisel3] def isFlipped(target: Data): Boolean = target match {
    case (element: Element) => element.binding.direction == Some(Direction.Input)
    case (vec: Vec[Data @unchecked]) => isFlipped(vec.sample_element)
    case (record: Record) => false
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
  // This is a bad API that punches through object boundaries.
  @deprecated("pending removal once all instances replaced", "chisel3")
  private[chisel3] def flatten: IndexedSeq[Element] = {
    this match {
      case elt: Aggregate => elt.getElements.toIndexedSeq flatMap {_.flatten}
      case elt: Element => IndexedSeq(elt)
      case elt => throwException(s"Cannot flatten type ${elt.getClass}")
    }
  }

  // Return ALL elements at root of this type.
  // Contasts with flatten, which returns just Bits
  // TODO: refactor away this, this is outside the scope of Data
  private[chisel3] def allElements: Seq[Element]

  private[core] def badConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    throwException(s"cannot connect ${this} and ${that}")
  private[chisel3] def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = {
    if (connectCompileOptions.checkSynthesizable) {
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
    } else {
      this legacyConnect that
    }
  }
  private[chisel3] def bulkConnect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = {
    if (connectCompileOptions.checkSynthesizable) {
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
    } else {
      this legacyConnect that
    }
  }

  /** Whether this Data has the same model ("data type") as that Data.
    * Data subtypes should overload this with checks against their own type.
    */
  private[core] def typeEquivalent(that: Data): Boolean

  private[chisel3] def lref: Node = Node(this)
  private[chisel3] def ref: Arg = if (isLit) litArg.get else lref
  private[chisel3] def toType: String
  private[core] def width: Width
  private[core] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit

  /** cloneType must be defined for any Chisel object extending Data.
    * It is responsible for constructing a basic copy of the object being cloned.
    * If cloneType needs to recursively clone elements of an object, it should call
    * the cloneType methods on those elements.
    * @return a copy of the object.
    */
  def cloneType: this.type

  /** chiselCloneType is called at the top-level of a clone chain.
    * It calls the client's cloneType() method to construct a basic copy of the object being cloned,
    * then performs any fixups required to reconstruct the appropriate core state of the cloned object.
    * @return a copy of the object with appropriate core state.
    */
  def chiselCloneType(implicit compileOptions: CompileOptions): this.type = {
    // TODO: refactor away allElements, handle this with Aggregate/Element match inside Bindings

    // Call the user-supplied cloneType method
    val clone = this.cloneType
    // In compatibility mode, simply return cloneType; otherwise, propagate
    // direction and flippedness.
    if (compileOptions.checkSynthesizable) {
      Data.setFirrtlDirection(clone, Data.getFirrtlDirection(this))
      //TODO(twigg): Do recursively for better error messages
      for((clone_elem, source_elem) <- clone.allElements zip this.allElements) {
        clone_elem.binding = UnboundBinding(source_elem.binding.direction)
        Data.setFirrtlDirection(clone_elem, Data.getFirrtlDirection(source_elem))
      }
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
  def fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): this.type = {
    val output = Wire(chiselCloneType).asInstanceOf[this.type]
    output.connectFromBits(that)
    output
  }

  /** Packs the value of this object as plain Bits.
    *
    * This performs the inverse operation of fromBits(Bits).
    */
  @deprecated("Best alternative, .asUInt()", "chisel3")
  def toBits(): UInt = do_asUInt(DeprecatedSourceInfo)

  /** Does a reinterpret cast of the bits in this node into the format that provides.
    * Returns a new Wire of that type. Does not modify existing nodes.
    *
    * x.asTypeOf(that) performs the inverse operation of x = that.toBits.
    *
    * @note bit widths are NOT checked, may pad or drop bits from input
    * @note that should have known widths
    */
  def asTypeOf[T <: Data](that: T): T = macro CompileOptionsTransform.thatArg

  def do_asTypeOf[T <: Data](that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val thatCloned = Wire(that.chiselCloneType)
    thatCloned.connectFromBits(this.asUInt())
    thatCloned
  }

  /** Assigns this node from Bits type. Internal implementation for asTypeOf.
    */
  private[core] def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit

  /** Reinterpret cast to UInt.
    *
    * @note value not guaranteed to be preserved: for example, a SInt of width
    * 3 and value -1 (0b111) would become an UInt with value 7
    * @note Aggregates are recursively packed with the first element appearing
    * in the least-significant bits of the result.
    */
  final def asUInt(): UInt = macro SourceInfoTransform.noArg

  def do_asUInt(implicit sourceInfo: SourceInfo): UInt

  // firrtlDirection is the direction we report to firrtl.
  // It maintains the user-specified value (as opposed to the "actual" or applied/propagated value).
  // NOTE: This should only be used for emitting acceptable firrtl.
  // The Element.dir should be used for any tests involving direction.
  private var firrtlDirection: Direction = Direction.Unspecified
  /** Default pretty printing */
  def toPrintable: Printable
}

object Wire {
  // No source info since Scala macros don't yet support named / default arguments.
  def apply[T <: Data](dummy: Int = 0, init: T)(implicit compileOptions: CompileOptions): T = {
    val model = (init.litArg match {
      // For e.g. Wire(init=UInt(0, k)), fix the Reg's width to k
      case Some(lit) if lit.forcedWidth => init.chiselCloneType
      case _ => init match {
        case init: Bits => init.cloneTypeWidth(Width())
        case init => init.chiselCloneType
      }
    }).asInstanceOf[T]
    apply(model, init)
  }

  // No source info since Scala macros don't yet support named / default arguments.
  def apply[T <: Data](t: T, init: T)(implicit compileOptions: CompileOptions): T = {
    implicit val noSourceInfo = UnlocatableSourceInfo
    val x = apply(t)
    Binding.checkSynthesizable(init, s"'init' ($init)")
    x := init
    x
  }

  def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val x = t.chiselCloneType

    // Bind each element of x to being a Wire
    Binding.bind(x, WireBinder(Builder.forcedModule), "Error: t")

    pushCommand(DefWire(sourceInfo, x))
    pushCommand(DefInvalid(sourceInfo, x.ref))

    x
  }
}

object Clock {
  def apply(): Clock = new Clock
  def apply(dir: Direction)(implicit compileOptions: CompileOptions): Clock = {
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
  private[chisel3] def toType = "Clock"

  private[core] def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass

  override def connect (that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: Clock => super.connect(that)(sourceInfo, connectCompileOptions)
    case _ => super.badConnect(that)(sourceInfo)
  }

  /** Not really supported */
  def toPrintable: Printable = PString("CLOCK")

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that
  }
}
