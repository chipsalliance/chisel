// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, UnlocatableSourceInfo, DeprecatedSourceInfo}
import chisel3.SourceInfoDoc
import chisel3.core.BiConnect.DontCareCantBeSink

/** User-specified directions.
  */
sealed abstract class SpecifiedDirection
object SpecifiedDirection {
  /** Default user direction, also meaning 'not-flipped'
    */
  case object Unspecified extends SpecifiedDirection
  /** Node and its children are forced as output
    */
  case object Output extends SpecifiedDirection
  /** Node and its children are forced as inputs
    */
  case object Input extends SpecifiedDirection
  /** Mainly for containers, children are flipped.
    */
  case object Flip extends SpecifiedDirection

  def flip(dir: SpecifiedDirection): SpecifiedDirection = dir match {
    case Unspecified => Flip
    case Flip => Unspecified
    case Output => Input
    case Input => Output
  }

  /** Returns the effective SpecifiedDirection of this node given the parent's effective SpecifiedDirection
    * and the user-specified SpecifiedDirection of this node.
    */
  def fromParent(parentDirection: SpecifiedDirection, thisDirection: SpecifiedDirection): SpecifiedDirection =
    (parentDirection, thisDirection) match {
      case (SpecifiedDirection.Output, _) => SpecifiedDirection.Output
      case (SpecifiedDirection.Input, _) => SpecifiedDirection.Input
      case (SpecifiedDirection.Unspecified, thisDirection) => thisDirection
      case (SpecifiedDirection.Flip, thisDirection) => SpecifiedDirection.flip(thisDirection)
    }
}

/** Resolved directions for both leaf and container nodes, only visible after
  * a node is bound (since higher-level specifications like Input and Output
  * can override directions).
  */
sealed abstract class ActualDirection

object ActualDirection {
  /** The object does not exist / is empty and hence has no direction
    */
  case object Empty extends ActualDirection

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

  def fromSpecified(direction: SpecifiedDirection): ActualDirection = direction match {
    case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip => ActualDirection.Unspecified
    case SpecifiedDirection.Output => ActualDirection.Output
    case SpecifiedDirection.Input => ActualDirection.Input
  }

  /** Determine the actual binding of a container given directions of its children.
    * Returns None in the case of mixed specified / unspecified directionality.
    */
  def fromChildren(childDirections: Set[ActualDirection], containerDirection: SpecifiedDirection):
      Option[ActualDirection] = {
    if (childDirections == Set()) {  // Sadly, Scala can't do set matching
      ActualDirection.fromSpecified(containerDirection) match {
        case ActualDirection.Unspecified => Some(ActualDirection.Empty)  // empty direction if relative / no direction
        case dir => Some(dir)  // use assigned direction if specified
      }
    } else if (childDirections == Set(ActualDirection.Unspecified)) {
      Some(ActualDirection.Unspecified)
    } else if (childDirections == Set(ActualDirection.Input)) {
      Some(ActualDirection.Input)
    } else if (childDirections == Set(ActualDirection.Output)) {
      Some(ActualDirection.Output)
    } else if (childDirections subsetOf
      Set(ActualDirection.Output, ActualDirection.Input,
        ActualDirection.Bidirectional(ActualDirection.Default),
        ActualDirection.Bidirectional(ActualDirection.Flipped))) {
      containerDirection match {
        case SpecifiedDirection.Unspecified => Some(ActualDirection.Bidirectional(ActualDirection.Default))
        case SpecifiedDirection.Flip => Some(ActualDirection.Bidirectional(ActualDirection.Flipped))
        case _ => throw new RuntimeException("Unexpected forced Input / Output")
      }
    } else {
      None
    }
  }
}

object debug {  // scalastyle:ignore object.name
  @chiselRuntimeDeprecated
  @deprecated("debug doesn't do anything in Chisel3 as no pruning happens in the frontend", "chisel3")
  def apply (arg: Data): Data = arg
}

/** Experimental hardware construction reflection API
  */
object DataMirror {
  def widthOf(target: Data): Width = target.width
  def specifiedDirectionOf(target: Data): SpecifiedDirection = target.specifiedDirection
  def directionOf(target: Data): ActualDirection = {
    requireIsHardware(target, "node requested directionality on")
    target.direction
  }

  // Returns the top-level module ports
  // TODO: maybe move to something like Driver or DriverUtils, since this is mainly for interacting
  // with compiled artifacts (vs. elaboration-time reflection)?
  def modulePorts(target: BaseModule): Seq[(String, Data)] = target.getChiselPorts

  // Returns all module ports with underscore-qualified names
  def fullModulePorts(target: BaseModule): Seq[(String, Data)] = {
    def getPortNames(name: String, data: Data): Seq[(String, Data)] = Seq(name -> data) ++ (data match {
      case _: Element => Seq()
      case r: Record => r.elements.toSeq flatMap { case (eltName, elt) => getPortNames(s"${name}_${eltName}", elt) }
      case v: Vec[_] => v.zipWithIndex flatMap { case (elt, index) => getPortNames(s"${name}_${index}", elt) }
    })
    modulePorts(target).flatMap { case (name, data) =>
      getPortNames(name, data).toList
    }
  }

  // Internal reflection-style APIs, subject to change and removal whenever.
  object internal { // scalastyle:ignore object.name
    def isSynthesizable(target: Data): Boolean = target.isSynthesizable
    // For those odd cases where you need to care about object reference and uniqueness
    def chiselTypeClone[T<:Data](target: Data): T = {
      target.cloneTypeFull.asInstanceOf[T]
    }
  }
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

    val filteredElts = elts.filter(_ != DontCare)
    require(!filteredElts.isEmpty, s"can't create $createdType with only DontCare inputs")

    if (filteredElts.head.isInstanceOf[Bits]) {
      val model: T = filteredElts reduce { (elt1: T, elt2: T) => ((elt1, elt2) match {
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
            s"can't create $createdType with heterogeneous types ${elt1.getClass} and ${elt2.getClass}")
      }).asInstanceOf[T] }
      model.cloneTypeFull
    }
    else {
      for (elt <- filteredElts.tail) {
        require(elt.getClass == filteredElts.head.getClass,
          s"can't create $createdType with heterogeneous types ${filteredElts.head.getClass} and ${elt.getClass}")
        require(elt typeEquivalent filteredElts.head,
          s"can't create $createdType with non-equivalent types ${filteredElts.head} and ${elt}")
      }
      filteredElts.head.cloneTypeFull
    }
  }
}

/** Returns the chisel type of a hardware object, allowing other hardware to be constructed from it.
  */
object chiselTypeOf {
  def apply[T <: Data](target: T): T = {
    requireIsHardware(target)
    target.cloneTypeFull.asInstanceOf[T]
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
    if (compileOptions.checkSynthesizable) {
      requireIsChiselType(source)
    }
    val out = source.cloneType.asInstanceOf[T]
    out.specifiedDirection = SpecifiedDirection.Input
    out
  }
}
object Output {
  def apply[T<:Data](source: T)(implicit compileOptions: CompileOptions): T = {
    if (compileOptions.checkSynthesizable) {
      requireIsChiselType(source)
    }
    val out = source.cloneType.asInstanceOf[T]
    out.specifiedDirection = SpecifiedDirection.Output
    out
  }
}
object Flipped {
  def apply[T<:Data](source: T)(implicit compileOptions: CompileOptions): T = {
    if (compileOptions.checkSynthesizable) {
      requireIsChiselType(source)
    }
    val out = source.cloneType.asInstanceOf[T]
    out.specifiedDirection = SpecifiedDirection.flip(source.specifiedDirection)
    out
  }
}

/** This forms the root of the type system for wire data types. The data value
  * must be representable as some number (need not be known at Chisel compile
  * time) of bits, and must have methods to pack / unpack structured data to /
  * from bits.
  *
  * @groupdesc Connect Utilities for connecting hardware components
  * @define coll data
  */
abstract class Data extends HasId with NamedComponent with SourceInfoDoc { // scalastyle:ignore number.of.methods
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
  // Note that the actual direction of this node can differ from child and parent specifiedDirection.
  private var _specifiedDirection: SpecifiedDirection = SpecifiedDirection.Unspecified
  private[chisel3] def specifiedDirection: SpecifiedDirection = _specifiedDirection
  private[core] def specifiedDirection_=(direction: SpecifiedDirection) = {
    if (_specifiedDirection != SpecifiedDirection.Unspecified) {
      this match {
        // Anything flies in compatibility mode
        case t: Record if !t.compileOptions.dontAssumeDirectionality =>
        case _ => throw Binding.RebindingException(s"Attempted reassignment of user-specified direction to $this")
      }
    }
    _specifiedDirection = direction
  }

  /** This overwrites a relative SpecifiedDirection with an explicit one, and is used to implement
    * the compatibility layer where, at the elements, Flip is Input and unspecified is Output.
    * DO NOT USE OUTSIDE THIS PURPOSE. THIS OPERATION IS DANGEROUS!
    */
  private[core] def _assignCompatibilityExplicitDirection: Unit = { // scalastyle:off method.name
    (this, _specifiedDirection) match {
      case (_: Analog, _) => // nothing to do
      case (_, SpecifiedDirection.Unspecified) => _specifiedDirection = SpecifiedDirection.Output
      case (_, SpecifiedDirection.Flip) => _specifiedDirection = SpecifiedDirection.Input
      case (_, SpecifiedDirection.Input | SpecifiedDirection.Output) => // nothing to do
    }
  }

  // Binding stores information about this node's position in the hardware graph.
  // This information is supplemental (more than is necessary to generate FIRRTL) and is used to
  // perform checks in Chisel, where more informative error messages are possible.
  private var _binding: Option[Binding] = None
  // Only valid after node is bound (synthesizable), crashes otherwise
  protected[core] def binding: Option[Binding] = _binding
  protected def binding_=(target: Binding) {
    if (_binding.isDefined) {
      throw Binding.RebindingException(s"Attempted reassignment of binding to $this")
    }
    _binding = Some(target)
  }

  // Similar to topBindingOpt except it explicitly excludes SampleElements which are bound but not
  // hardware
  private[core] final def isSynthesizable: Boolean = _binding.map {
    case ChildBinding(parent) => parent.isSynthesizable
    case _: TopBinding => true
    case _: SampleElementBinding[_] => false
  }.getOrElse(false)

  private[core] def topBindingOpt: Option[TopBinding] = _binding.flatMap {
    case ChildBinding(parent) => parent.topBindingOpt
    case bindingVal: TopBinding => Some(bindingVal)
    case SampleElementBinding(parent) => parent.topBindingOpt
  }

  private[core] def topBinding: TopBinding = topBindingOpt.get

  /** Binds this node to the hardware graph.
    * parentDirection is the direction of the parent node, or Unspecified (default) if the target
    * node is the top-level.
    * binding and direction are valid after this call completes.
    */
  private[chisel3] def bind(target: Binding, parentDirection: SpecifiedDirection = SpecifiedDirection.Unspecified)

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

  // User-friendly representation of the binding as a helper function for toString.
  // Provides a unhelpful fallback for literals, which should have custom rendering per
  // Data-subtype.
  // TODO Is this okay for sample_element? It *shouldn't* be visible to users
  protected def bindingToString: String = topBindingOpt match {
    case None => ""
    case Some(OpBinding(enclosure)) => s"(OpResult in ${enclosure.desiredName})"
    case Some(MemoryPortBinding(enclosure)) => s"(MemPort in ${enclosure.desiredName})"
    case Some(PortBinding(enclosure)) if !enclosure.isClosed => s"(IO in unelaborated ${enclosure.desiredName})"
    case Some(PortBinding(enclosure)) if enclosure.isClosed =>
      DataMirror.fullModulePorts(enclosure).find(_._2 eq this) match {
        case Some((name, _)) => s"(IO $name in ${enclosure.desiredName})"
        case None => s"(IO (unknown) in ${enclosure.desiredName})"
      }
    case Some(RegBinding(enclosure)) => s"(Reg in ${enclosure.desiredName})"
    case Some(WireBinding(enclosure)) => s"(Wire in ${enclosure.desiredName})"
    case Some(DontCareBinding()) => s"(DontCare)"
    case Some(ElementLitBinding(litArg)) => s"(unhandled literal)"
    case Some(BundleLitBinding(litMap)) => s"(unhandled bundle literal)"
  }

  // Return ALL elements at root of this type.
  // Contasts with flatten, which returns just Bits
  // TODO: refactor away this, this is outside the scope of Data
  private[chisel3] def allElements: Seq[Element]

  private[core] def badConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    throwException(s"cannot connect ${this} and ${that}")
  private[chisel3] def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = { // scalastyle:ignore line.size.limit
    if (connectCompileOptions.checkSynthesizable) {
      requireIsHardware(this, "data to be connected")
      requireIsHardware(that, "data to be connected")
      this.topBinding match {
        case _: ReadOnlyBinding => throwException(s"Cannot reassign to read-only $this")
        case _ =>  // fine
      }
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
  private[chisel3] def bulkConnect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = { // scalastyle:ignore line.size.limit
    if (connectCompileOptions.checkSynthesizable) {
      requireIsHardware(this, s"data to be bulk-connected")
      requireIsHardware(that, s"data to be bulk-connected")
      (this.topBinding, that.topBinding) match {
        case (_: ReadOnlyBinding, _: ReadOnlyBinding) => throwException(s"Both $this and $that are read-only")
        // DontCare cannot be a sink (LHS)
        case (_: DontCareBinding, _) => throw DontCareCantBeSink
        case _ =>  // fine
      }
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

  // Internal API: returns a ref that can be assigned to, if consistent with the binding
  private[chisel3] def lref: Node = {
    requireIsHardware(this)
    topBindingOpt match {
      case Some(binding: ReadOnlyBinding) => throwException(s"internal error: attempted to generate LHS ref to ReadOnlyBinding $binding") // scalastyle:ignore line.size.limit
      case Some(binding: TopBinding) => Node(this)
      case opt => throwException(s"internal error: unknown binding $opt in generating LHS ref")
    }
  }


  // Internal API: returns a ref, if bound. Literals should override this as needed.
  private[chisel3] def ref: Arg = {
    requireIsHardware(this)
    topBindingOpt match {
      case Some(binding: LitBinding) => throwException(s"internal error: can't handle literal binding $binding")
      case Some(binding: TopBinding) => Node(this)
      case opt => throwException(s"internal error: unknown binding $opt in generating LHS ref")
    }
  }

  private[chisel3] def width: Width
  private[core] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit

  /** Internal API; Chisel users should look at chisel3.chiselTypeOf(...).
    *
    * cloneType must be defined for any Chisel object extending Data.
    * It is responsible for constructing a basic copy of the object being cloned.
    *
    * @return a copy of the object.
    */
  def cloneType: this.type

  /** Internal API; Chisel users should look at chisel3.chiselTypeOf(...).
    *
    * Returns a copy of this data type, with hardware bindings (if any) removed.
    * Directionality data is still preserved.
    */
  private[chisel3] def cloneTypeFull: this.type = {
    val clone = this.cloneType.asInstanceOf[this.type]  // get a fresh object, without bindings
    // Only the top-level direction needs to be fixed up, cloneType should do the rest
    clone.specifiedDirection = specifiedDirection
    clone
  }

  /** Connect this $coll to that $coll mono-directionally and element-wise.
    *
    * This uses the [[MonoConnect]] algorithm.
    *
    * @param that the $coll to connect to
    * @group Connect
    */
  final def := (that: Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit = this.connect(that)(sourceInfo, connectionCompileOptions) // scalastyle:ignore line.size.limit

  /** Connect this $coll to that $coll bi-directionally and element-wise.
    *
    * This uses the [[BiConnect]] algorithm.
    *
    * @param that the $coll to connect to
    * @group Connect
    */
  final def <> (that: Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit = this.bulkConnect(that)(sourceInfo, connectionCompileOptions) // scalastyle:ignore line.size.limit

  @chiselRuntimeDeprecated
  @deprecated("litArg is deprecated, use litOption or litTo*Option", "chisel3.2")
  def litArg(): Option[LitArg] = topBindingOpt match {
    case Some(ElementLitBinding(litArg)) => Some(litArg)
    case Some(BundleLitBinding(litMap)) => None  // this API does not support Bundle literals
    case _ => None
  }

  def isLit(): Boolean = litArg.isDefined

  /**
   * If this is a literal that is representable as bits, returns the value as a BigInt.
   * If not a literal, or not representable as bits (for example, is or contains Analog), returns None.
   */
  def litOption(): Option[BigInt]

  /**
   * Returns the literal value if this is a literal that is representable as bits, otherwise crashes.
   */
  def litValue(): BigInt = litOption.get

  /** Returns the width, in bits, if currently known. */
  final def getWidth: Int =
    if (isWidthKnown) width.get else throwException(s"Width of $this is unknown!")
  /** Returns whether the width is currently known. */
  final def isWidthKnown: Boolean = width.known
  /** Returns Some(width) if the width is known, else None. */
  final def widthOption: Option[Int] = if (isWidthKnown) Some(getWidth) else None

  /** Packs the value of this object as plain Bits.
    *
    * This performs the inverse operation of fromBits(Bits).
    */
  @chiselRuntimeDeprecated
  @deprecated("Best alternative, .asUInt()", "chisel3")
  def toBits(implicit compileOptions: CompileOptions): UInt = do_asUInt(DeprecatedSourceInfo, compileOptions)

  /** Does a reinterpret cast of the bits in this node into the format that provides.
    * Returns a new Wire of that type. Does not modify existing nodes.
    *
    * x.asTypeOf(that) performs the inverse operation of x := that.toBits.
    *
    * @note bit widths are NOT checked, may pad or drop bits from input
    * @note that should have known widths
    */
  def asTypeOf[T <: Data](that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_asTypeOf[T <: Data](that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val thatCloned = Wire(that.cloneTypeFull)
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

  /** @group SourceInfoTransformMacro */
  def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

  /** Default pretty printing */
  def toPrintable: Printable
}

trait WireFactory {
  /** Construct a [[Wire]] from a type template
    * @param t The template from which to construct this wire
    */
  def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(t, "wire type")
    }
    val x = t.cloneTypeFull

    // Bind each element of x to being a Wire
    x.bind(WireBinding(Builder.forcedUserModule))

    pushCommand(DefWire(sourceInfo, x))
    if (!compileOptions.explicitInvalidate) {
      pushCommand(DefInvalid(sourceInfo, x.ref))
    }

    x
  }
}

/** Utility for constructing hardware wires
  *
  * The width of a `Wire` (inferred or not) is copied from the type template
  * {{{
  * val w0 = Wire(UInt()) // width is inferred
  * val w1 = Wire(UInt(8.W)) // width is set to 8
  *
  * val w2 = Wire(Vec(4, UInt())) // width is inferred
  * val w3 = Wire(Vec(4, UInt(8.W))) // width of each element is set to 8
  *
  * class MyBundle {
  *   val unknown = UInt()
  *   val known   = UInt(8.W)
  * }
  * val w4 = Wire(new MyBundle)
  * // Width of w4.unknown is inferred
  * // Width of w4.known is set to 8
  * }}}
  *
  */
object Wire extends WireFactory


/** Utility for constructing hardware wires with a default connection
  *
  * The two forms of `WireDefault` differ in how the type and width of the resulting [[Wire]] are
  * specified.
  *
  * ==Single Argument==
  * The single argument form uses the argument to specify both the type and default connection. For
  * non-literal [[Bits]], the width of the [[Wire]] will be inferred. For literal [[Bits]] and all
  * non-Bits arguments, the type will be copied from the argument. See the following examples for
  * more details:
  *
  * 1. Literal [[Bits]] initializer: width will be set to match
  * {{{
  * val w1 = WireDefault(1.U) // width will be inferred to be 1
  * val w2 = WireDefault(1.U(8.W)) // width is set to 8
  * }}}
  *
  * 2. Non-Literal [[Element]] initializer - width will be inferred
  * {{{
  * val x = Wire(UInt())
  * val y = Wire(UInt(8.W))
  * val w1 = WireDefault(x) // width will be inferred
  * val w2 = WireDefault(y) // width will be inferred
  * }}}
  *
  * 3. [[Aggregate]] initializer - width will be set to match the aggregate
  *
  * {{{
  * class MyBundle {
  *   val unknown = UInt()
  *   val known   = UInt(8.W)
  * }
  * val w1 = Wire(new MyBundle)
  * val w2 = WireDefault(w1)
  * // Width of w2.unknown is inferred
  * // Width of w2.known is set to 8
  * }}}
  *
  * ==Double Argument==
  * The double argument form allows the type of the [[Wire]] and the default connection to be
  * specified independently.
  *
  * The width inference semantics for `WireDefault` with two arguments match those of [[Wire]]. The
  * first argument to `WireDefault` is the type template which defines the width of the `Wire` in
  * exactly the same way as the only argument to [[Wire]].
  *
  * More explicitly, you can reason about `WireDefault` with multiple arguments as if it were defined
  * as:
  * {{{
  * def WireDefault[T <: Data](t: T, init: T): T = {
  *   val x = Wire(t)
  *   x := init
  *   x
  * }
  * }}}
  *
  * @note The `Default` in `WireDefault` refers to a `default` connection. This is in contrast to
  * [[RegInit]] where the `Init` refers to a value on reset.
  */
object WireDefault {

  private def applyImpl[T <: Data](t: T, init: Data)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = { // scalastyle:ignore line.size.limit
    implicit val noSourceInfo = UnlocatableSourceInfo
    val x = Wire(t)
    requireIsHardware(init, "wire initializer")
    x := init
    x
  }

  /** Construct a [[Wire]] with a type template and a [[chisel3.DontCare]] default
    * @param t The type template used to construct this [[Wire]]
    * @param init The default connection to this [[Wire]], can only be [[DontCare]]
    * @note This is really just a specialized form of `apply[T <: Data](t: T, init: T): T` with [[DontCare]] as `init`
    */
  def apply[T <: Data](t: T, init: DontCare.type)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = { // scalastyle:ignore line.size.limit
    applyImpl(t, init)
  }

  /** Construct a [[Wire]] with a type template and a default connection
    * @param t The type template used to construct this [[Wire]]
    * @param init The hardware value that will serve as the default value
    */
  def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    applyImpl(t, init)
  }

  /** Construct a [[Wire]] with a default connection
    * @param init The hardware value that will serve as a type template and default value
    */
  def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val model = (init match {
      // If init is a literal without forced width OR any non-literal, let width be inferred
      case init: Bits if !init.litIsForcedWidth.getOrElse(false) => init.cloneTypeWidth(Width())
      case _ => init.cloneTypeFull
    }).asInstanceOf[T]
    apply(model, init)
  }
}

/** RHS (source) for Invalidate API.
  * Causes connection logic to emit a DefInvalid when connected to an output port (or wire).
  */
private[chisel3] object DontCare extends Element {
  // This object should be initialized before we execute any user code that refers to it,
  //  otherwise this "Chisel" object will end up on the UserModule's id list.
  // We make it private to chisel3 so it has to be accessed through the package object.

  private[chisel3] override val width: Width = UnknownWidth()

  bind(DontCareBinding(), SpecifiedDirection.Output)
  override def cloneType: this.type = DontCare

  override def toString: String = "DontCare()"

  override def litOption: Option[BigInt] = None

  def toPrintable: Printable = PString("DONTCARE")

  private[core] def connectFromBits(that: chisel3.core.Bits)(implicit sourceInfo:  SourceInfo, compileOptions: CompileOptions): Unit = { // scalastyle:ignore line.size.limit
    Builder.error("connectFromBits: DontCare cannot be a connection sink (LHS)")
  }

  def do_asUInt(implicit sourceInfo: chisel3.internal.sourceinfo.SourceInfo, compileOptions: CompileOptions): chisel3.core.UInt = { // scalastyle:ignore line.size.limit
    Builder.error("DontCare does not have a UInt representation")
    0.U
  }
  // DontCare's only match themselves.
  private[core] def typeEquivalent(that: chisel3.core.Data): Boolean = that == DontCare
}
