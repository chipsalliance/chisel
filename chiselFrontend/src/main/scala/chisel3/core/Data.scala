// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._

/** User-specified directions.
  */
sealed abstract class UserDirection
object UserDirection {
  /** Default user direction, also meaning 'not-flipped'
    */
  case object Unspecified extends UserDirection
  /** Node and its children are forced as output
    */
  case object Output extends UserDirection
  /** Node and its children are forced as inputs
    */
  case object Input extends UserDirection
  /** Mainly for containers, children are flipped.
    */
  case object Flip extends UserDirection

  def flip(dir: UserDirection) = dir match {
    case Unspecified => Flip
    case Flip => Unspecified
    case Output => Input
    case Input => Output
  }

  /** Returns the effective UserDirection of this node given the parent's effective UserDirection
    * and the user-specified UserDirection of this node.
    */
  def fromParent(parentDirection: UserDirection, thisDirection: UserDirection) =
    (parentDirection, thisDirection) match {
      case (UserDirection.Output, _) => UserDirection.Output
      case (UserDirection.Input, _) => UserDirection.Input
      case (UserDirection.Unspecified, thisDirection) => thisDirection
      case (UserDirection.Flip, thisDirection) => UserDirection.flip(thisDirection)
    }
}

/** Resolved directions for both leaf and container nodes, only visible after
  * a node is bound (since higher-level specifications like Input and Output
  * can override directions).
  */
sealed abstract class ActualDirection

object ActualDirection {
  /** Undirectioned, struct-like
    */
  case object Unspecified extends ActualDirection
  /** Output element, or container with all outputs (even if forced)
    */
  case object Output extends ActualDirection
  /** Input element, or container with all inputs (even if forced)
    */
  case object Input extends ActualDirection

  sealed abstract class BidirectionalDirection
  case object Default extends BidirectionalDirection
  case object Flipped extends BidirectionalDirection

  case class Bidirectional(dir: BidirectionalDirection) extends ActualDirection
}

@deprecated("debug doesn't do anything in Chisel3 as no pruning happens in the frontend", "chisel3")
object debug {  // scalastyle:ignore object.name
  def apply (arg: Data): Data = arg
}

/** Experimental hardware construction reflection API
  */
object DataMirror {
  def widthOf(target: Data): Width = target.width
  def userDirectionOf(target: Data): UserDirection = target.userDirection
  def directionOf(target: Data): ActualDirection = {
    requireIsHardware(target, "node requested directionality on")
    target.direction
  }
  // TODO: really not a reflection-style API, but a workaround for dir in the compatibility package
  def isSynthesizable(target: Data) = target.hasBinding
}

/** Creates a clone of the super-type of the input elements. Super-type is defined as:
  * - for Bits type of the same class: the cloned type of the largest width
  * - Bools are treated as UInts
  * - For other types of the same class are are the same: clone of any of the elements
  * - Otherwise: fail
  */
//scalastyle:off cyclomatic.complexity
private[core] object cloneSupertype {
  def apply[T <: Data](elts: Seq[T], createdType: String)(implicit sourceInfo: SourceInfo,
                                                          compileOptions: CompileOptions): T = {
    require(!elts.isEmpty, s"can't create $createdType with no inputs")

    if (elts forall {_.isInstanceOf[Bits]}) {
      val model: T = elts reduce { (elt1: T, elt2: T) => ((elt1, elt2) match {
        case (elt1: Bool, elt2: Bool) => elt1
        case (elt1: Bool, elt2: UInt) => elt2  // TODO: what happens with zero width UInts?
        case (elt1: UInt, elt2: Bool) => elt1  // TODO: what happens with zero width UInts?
        case (elt1: UInt, elt2: UInt) =>
          // TODO: perhaps redefine Widths to allow >= op?
          if (elt1.width == (elt1.width max elt2.width)) elt1 else elt2
        case (elt1: SInt, elt2: SInt) => if (elt1.width == (elt1.width max elt2.width)) elt1 else elt2
        case (elt1: FixedPoint, elt2: FixedPoint) => {
          (elt1.binaryPoint, elt2.binaryPoint, elt1.width, elt2.width) match {
            case (KnownBinaryPoint(bp1), KnownBinaryPoint(bp2), KnownWidth(w1), KnownWidth(w2)) =>
              val maxBinaryPoint = bp1 max bp2
              val maxIntegerWidth = (w1 - bp1) max (w2 - bp2)
              FixedPoint((maxIntegerWidth + maxBinaryPoint).W, (maxBinaryPoint).BP)
            case (KnownBinaryPoint(bp1), KnownBinaryPoint(bp2), _, _) =>
              FixedPoint(Width(), (bp1 max bp2).BP)
            case _ => FixedPoint()
          }
        }
        case (elt1, elt2) =>
          throw new AssertionError(
            s"can't create $createdType with heterogeneous Bits types ${elt1.getClass} and ${elt2.getClass}")
      }).asInstanceOf[T] }
      model.chiselCloneType
    }
    else {
      for (elt <- elts.tail) {
        require(elt.getClass == elts.head.getClass,
          s"can't create $createdType with heterogeneous types ${elts.head.getClass} and ${elt.getClass}")
        require(elt typeEquivalent elts.head,
          s"can't create $createdType with non-equivalent types ${elts.head} and ${elt}")
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
  def apply[T<:Data](source: T): T = {
    val out = source.cloneType
    out.userDirection = UserDirection.Input
    out
  }
}
object Output {
  def apply[T<:Data](source: T): T = {
    val out = source.cloneType
    out.userDirection = UserDirection.Output
    out
  }
}
object Flipped {
  def apply[T<:Data](source: T): T = {
    val out = source.cloneType
    out.userDirection = UserDirection.flip(source.userDirection)
    out
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

  // User-specified direction, local at this node only.
  // Note that the actual direction of this node can differ from child and parent userDirection.
  private var _userDirection: UserDirection = UserDirection.Unspecified
  private[chisel3] def userDirection: UserDirection = _userDirection
  private[core] def userDirection_=(direction: UserDirection) = {
    if (_userDirection != UserDirection.Unspecified) {
      this match {
        // Anything flies in compatibility mode
        case t: Record if !t.compileOptions.dontAssumeDirectionality =>
        case _ => throw Binding.RebindingException(s"Attempted reassignment of user direction to $this")
      }
    }
    _userDirection = direction
  }

  /** This overwrites a relative UserDirection with an explicit one, and is used to implement
    * the compatibility layer where, at the elements, Flip is Input and unspecified is Output.
    * DO NOT USE OUTSIDE THIS PURPOSE. THIS OPERATION IS DANGEROUS!
    */
  private[core] def _assignCompatibilityExplicitDirection: Unit = {
    _userDirection match {
      case UserDirection.Unspecified => _userDirection = UserDirection.Output
      case UserDirection.Flip => _userDirection = UserDirection.Input
      case UserDirection.Input | UserDirection.Output => // nothing to do
    }
  }

  // Binding stores information about this node's position in the hardware graph.
  // This information is supplemental (more than is necessary to generate FIRRTL) and is used to
  // perform checks in Chisel, where more informative error messages are possible.
  private var _binding: Option[Binding] = None
  private[core] def hasBinding = _binding.isDefined
  // Only valid after node is bound (synthesizable), crashes otherwise
  private[core] def binding = _binding.get
  protected def binding_=(target: Binding) {
    if (_binding.isDefined) {
      throw Binding.RebindingException(s"Attempted reassignment of binding to $this")
    }
    _binding = Some(target)
  }

  private[core] def topBinding: TopBinding = {
    binding match {
      case ChildBinding(parent) => parent.topBinding
      case topBinding: TopBinding => topBinding
    }
  }

  /** Binds this node to the hardware graph.
    * parentDirection is the direction of the parent node, or Unspecified (default) if the target
    * node is the top-level.
    * binding and direction are valid after this call completes.
    */
  private[chisel3] def bind(target: Binding, parentDirection: UserDirection = UserDirection.Unspecified)

  // Both _direction and _resolvedUserDirection are saved versions of computed variables (for
  // efficiency, avoid expensive recomputation of frequent operations).
  // Both are only valid after binding is set.

  // Direction of this node, accounting for parents (force Input / Output) and children.
  private var _direction: Option[ActualDirection] = None

  private[chisel3] def direction: ActualDirection = _direction.get
  private[core] def direction_=(actualDirection: ActualDirection) {
    if (_direction.isDefined) {
      throw Binding.RebindingException(s"Attempted reassignment of resolved direction to $this")
    }
    _direction = Some(actualDirection)
  }

  // Return ALL elements at root of this type.
  // Contasts with flatten, which returns just Bits
  // TODO: refactor away this, this is outside the scope of Data
  private[chisel3] def allElements: Seq[Element]

  private[core] def badConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    throwException(s"cannot connect ${this} and ${that}")
  private[chisel3] def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = {
    if (connectCompileOptions.checkSynthesizable) {
      requireIsHardware(this, "data to be connected")
      requireIsHardware(that, "data to be connected")
      try {
        MonoConnect.connect(sourceInfo, connectCompileOptions, this, that, Builder.forcedUserModule)
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
      requireIsHardware(this, s"data to be bulk-connected")
      requireIsHardware(that, s"data to be bulk-connected")
      try {
        BiConnect.connect(sourceInfo, connectCompileOptions, this, that, Builder.forcedUserModule)
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
  private[chisel3] def width: Width
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
  def chiselCloneType: this.type = {
    val clone = this.cloneType  // get a fresh object, without bindings
    // Only the top-level direction needs to be fixed up, cloneType should do the rest
    clone.userDirection = userDirection
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

  /** Packs the value of this object as plain Bits.
    *
    * This performs the inverse operation of fromBits(Bits).
    */
  @deprecated("Best alternative, .asUInt()", "chisel3")
  def toBits(implicit compileOptions: CompileOptions): UInt = do_asUInt(DeprecatedSourceInfo, compileOptions)

  /** Does a reinterpret cast of the bits in this node into the format that provides.
    * Returns a new Wire of that type. Does not modify existing nodes.
    *
    * x.asTypeOf(that) performs the inverse operation of x = that.toBits.
    *
    * @note bit widths are NOT checked, may pad or drop bits from input
    * @note that should have known widths
    */
  def asTypeOf[T <: Data](that: T): T = macro SourceInfoTransform.thatArg

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

  def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

  /** Default pretty printing */
  def toPrintable: Printable
}

object Wire {
  // No source info since Scala macros don't yet support named / default arguments.
  def apply[T <: Data](dummy: Int = 0, init: T)(implicit compileOptions: CompileOptions): T = {
    val model = (init.litArg match {
      // For e.g. Wire(init=0.U(k.W)), fix the Reg's width to k
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
    requireIsHardware(init, "wire initializer")
    x := init
    x
  }

  def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val x = t.chiselCloneType

    // Bind each element of x to being a Wire
    x.bind(WireBinding(Builder.forcedUserModule))

    pushCommand(DefWire(sourceInfo, x))
    pushCommand(DefInvalid(sourceInfo, x.ref))

    x
  }
}
