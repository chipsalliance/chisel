// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.dataview.reify

import scala.language.experimental.macros
import chisel3.experimental.{Analog, BaseModule, DataMirror, EnumType, FixedPoint, Interval}
import chisel3.internal.Builder.pushCommand
import chisel3.internal._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, UnlocatableSourceInfo}

import scala.collection.immutable.LazyList // Needed for 2.12 alias
import scala.reflect.ClassTag
import scala.util.Try

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
    case Flip        => Unspecified
    case Output      => Input
    case Input       => Output
  }

  /** Returns the effective SpecifiedDirection of this node given the parent's effective SpecifiedDirection
    * and the user-specified SpecifiedDirection of this node.
    */
  def fromParent(parentDirection: SpecifiedDirection, thisDirection: SpecifiedDirection): SpecifiedDirection =
    (parentDirection, thisDirection) match {
      case (SpecifiedDirection.Output, _)                  => SpecifiedDirection.Output
      case (SpecifiedDirection.Input, _)                   => SpecifiedDirection.Input
      case (SpecifiedDirection.Unspecified, thisDirection) => thisDirection
      case (SpecifiedDirection.Flip, thisDirection)        => SpecifiedDirection.flip(thisDirection)
    }

  private[chisel3] def specifiedDirection[T <: Data](
    source: => T
  )(dir:    T => SpecifiedDirection
  )(
    implicit compileOptions: CompileOptions
  ): T = {
    val prevId = Builder.idGen.value
    val data = source // evaluate source once (passed by name)
    if (compileOptions.checkSynthesizable) {
      requireIsChiselType(data)
    }
    val out = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    out.specifiedDirection = dir(out)
    out
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
    case SpecifiedDirection.Output                                => ActualDirection.Output
    case SpecifiedDirection.Input                                 => ActualDirection.Input
  }

  /** Determine the actual binding of a container given directions of its children.
    * Returns None in the case of mixed specified / unspecified directionality.
    */
  def fromChildren(
    childDirections:    Set[ActualDirection],
    containerDirection: SpecifiedDirection
  ): Option[ActualDirection] = {
    if (childDirections == Set()) { // Sadly, Scala can't do set matching
      ActualDirection.fromSpecified(containerDirection) match {
        case ActualDirection.Unspecified => Some(ActualDirection.Empty) // empty direction if relative / no direction
        case dir                         => Some(dir) // use assigned direction if specified
      }
    } else if (childDirections == Set(ActualDirection.Unspecified)) {
      Some(ActualDirection.Unspecified)
    } else if (childDirections == Set(ActualDirection.Input)) {
      Some(ActualDirection.Input)
    } else if (childDirections == Set(ActualDirection.Output)) {
      Some(ActualDirection.Output)
    } else if (
      childDirections.subsetOf(
        Set(
          ActualDirection.Output,
          ActualDirection.Input,
          ActualDirection.Bidirectional(ActualDirection.Default),
          ActualDirection.Bidirectional(ActualDirection.Flipped)
        )
      )
    ) {
      containerDirection match {
        case SpecifiedDirection.Unspecified => Some(ActualDirection.Bidirectional(ActualDirection.Default))
        case SpecifiedDirection.Flip        => Some(ActualDirection.Bidirectional(ActualDirection.Flipped))
        case _                              => throw new RuntimeException("Unexpected forced Input / Output")
      }
    } else {
      None
    }
  }
}

/** Creates a clone of the super-type of the input elements. Super-type is defined as:
  * - for Bits type of the same class: the cloned type of the largest width
  * - Bools are treated as UInts
  * - For other types of the same class are are the same: clone of any of the elements
  * - Otherwise: fail
  */
private[chisel3] object cloneSupertype {
  def apply[T <: Data](
    elts:        Seq[T],
    createdType: String
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
    require(!elts.isEmpty, s"can't create $createdType with no inputs")

    val filteredElts = elts.filter(_ != DontCare)
    require(!filteredElts.isEmpty, s"can't create $createdType with only DontCare inputs")

    if (filteredElts.head.isInstanceOf[Bits]) {
      val model: T = filteredElts.reduce { (elt1: T, elt2: T) =>
        ((elt1, elt2) match {
          case (elt1: Bool, elt2: Bool) => elt1
          case (elt1: Bool, elt2: UInt) => elt2 // TODO: what happens with zero width UInts?
          case (elt1: UInt, elt2: Bool) => elt1 // TODO: what happens with zero width UInts?
          case (elt1: UInt, elt2: UInt) =>
            // TODO: perhaps redefine Widths to allow >= op?
            if (elt1.width == (elt1.width.max(elt2.width))) elt1 else elt2
          case (elt1: SInt, elt2: SInt) => if (elt1.width == (elt1.width.max(elt2.width))) elt1 else elt2
          case (elt1: FixedPoint, elt2: FixedPoint) => {
            (elt1.binaryPoint, elt2.binaryPoint, elt1.width, elt2.width) match {
              case (KnownBinaryPoint(bp1), KnownBinaryPoint(bp2), KnownWidth(w1), KnownWidth(w2)) =>
                val maxBinaryPoint = bp1.max(bp2)
                val maxIntegerWidth = (w1 - bp1).max(w2 - bp2)
                FixedPoint((maxIntegerWidth + maxBinaryPoint).W, (maxBinaryPoint).BP)
              case (KnownBinaryPoint(bp1), KnownBinaryPoint(bp2), _, _) =>
                FixedPoint(Width(), (bp1.max(bp2)).BP)
              case _ => FixedPoint()
            }
          }
          case (elt1: Interval, elt2: Interval) =>
            val range = if (elt1.range.width == elt1.range.width.max(elt2.range.width)) elt1.range else elt2.range
            Interval(range)
          case (elt1, elt2) =>
            throw new AssertionError(
              s"can't create $createdType with heterogeneous types ${elt1.getClass} and ${elt2.getClass}"
            )
        }).asInstanceOf[T]
      }
      model.cloneTypeFull
    } else {
      for (elt <- filteredElts.tail) {
        require(
          elt.getClass == filteredElts.head.getClass,
          s"can't create $createdType with heterogeneous types ${filteredElts.head.getClass} and ${elt.getClass}"
        )
        require(
          elt.typeEquivalent(filteredElts.head),
          s"can't create $createdType with non-equivalent types ${filteredElts.head} and ${elt}"
        )
      }
      filteredElts.head.cloneTypeFull
    }
  }
}

// Returns pairs of all fields, element-level and containers, in a Record and their path names
private[chisel3] object getRecursiveFields {
  def noPath(data: Data): Seq[Data] = data match {
    case data: Record =>
      data.elements.map {
        case (_, fieldData) =>
          getRecursiveFields.noPath(fieldData)
      }.fold(Seq(data)) {
        _ ++ _
      }
    case data: Vec[_] =>
      data.getElements.zipWithIndex.map {
        case (fieldData, fieldIndex) =>
          getRecursiveFields.noPath(fieldData)
      }.fold(Seq(data)) {
        _ ++ _
      }
    case data: Element => Seq(data)
  }
  def apply(data: Data, path: String): Seq[(Data, String)] = data match {
    case data: Record =>
      data.elements.map {
        case (fieldName, fieldData) =>
          getRecursiveFields(fieldData, s"$path.$fieldName")
      }.fold(Seq(data -> path)) {
        _ ++ _
      }
    case data: Vec[_] =>
      data.elementsIterator.zipWithIndex.map {
        case (fieldData, fieldIndex) =>
          getRecursiveFields(fieldData, path = s"$path($fieldIndex)")
      }.fold(Seq(data -> path)) {
        _ ++ _
      }
    case data: Element => Seq(data -> path)
  }

  def lazily(data: Data, path: String): Seq[(Data, String)] = data match {
    case data: Record =>
      LazyList(data -> path) ++
        data.elements.view.flatMap {
          case (fieldName, fieldData) =>
            getRecursiveFields(fieldData, s"$path.$fieldName")
        }
    case data: Vec[_] =>
      LazyList(data -> path) ++
        data.elementsIterator.zipWithIndex.flatMap {
          case (fieldData, fieldIndex) =>
            getRecursiveFields(fieldData, path = s"$path($fieldIndex)")
        }
    case data: Element => LazyList(data -> path)
  }
  def lazilyNoPath(data: Data): Seq[Data] = data match {
    case data: Record =>
      LazyList(data) ++
        data.elements.view.flatMap {
          case (fieldName, fieldData) =>
            getRecursiveFields.lazilyNoPath(fieldData)
        }
    case data: Vec[_] =>
      LazyList(data) ++
        data.getElements.view.flatMap {
          case (fieldData) =>
            getRecursiveFields.lazilyNoPath(fieldData)
        }
    case data: Element => LazyList(data)
  }
}

// Returns pairs of corresponding fields between two Records of the same type
// TODO it seems wrong that Elements are checked for typeEquivalence in Bundle and Vec lit creation
private[chisel3] object getMatchedFields {
  def apply(x: Data, y: Data): Seq[(Data, Data)] = (x, y) match {
    case (x: Element, y: Element) =>
      require(x.typeEquivalent(y))
      Seq(x -> y)
    case (x: Record, y: Record) =>
      (x.elements
        .zip(y.elements))
        .map {
          case ((xName, xElt), (yName, yElt)) =>
            require(
              xName == yName,
              s"$xName != $yName, ${x.elements}, ${y.elements}, $x, $y"
            ) // assume fields returned in same, deterministic order
            getMatchedFields(xElt, yElt)
        }
        .fold(Seq(x -> y)) {
          _ ++ _
        }
    case (x: Vec[_], y: Vec[_]) =>
      (x.elementsIterator
        .zip(y.elementsIterator))
        .map {
          case (xElt, yElt) =>
            getMatchedFields(xElt, yElt)
        }
        .fold(Seq(x -> y)) {
          _ ++ _
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
  def apply[T <: Data](source: => T)(implicit compileOptions: CompileOptions): T = {
    SpecifiedDirection.specifiedDirection(source)(_ => SpecifiedDirection.Input)
  }
}
object Output {
  def apply[T <: Data](source: => T)(implicit compileOptions: CompileOptions): T = {
    SpecifiedDirection.specifiedDirection(source)(_ => SpecifiedDirection.Output)
  }
}

object Flipped {
  def apply[T <: Data](source: => T)(implicit compileOptions: CompileOptions): T = {
    SpecifiedDirection.specifiedDirection(source)(x => SpecifiedDirection.flip(x.specifiedDirection))
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
abstract class Data extends HasId with NamedComponent with SourceInfoDoc {
  // This is a bad API that punches through object boundaries.
  private[chisel3] def flatten: IndexedSeq[Element] = {
    this match {
      case elt: Aggregate => elt.elementsIterator.toIndexedSeq.flatMap { _.flatten }
      case elt: Element   => IndexedSeq(elt)
      case elt => throwException(s"Cannot flatten type ${elt.getClass}")
    }
  }

  // must clone a data if
  // * it has a binding
  // * its id is older than prevId (not "freshly created")
  // * it is a bundle with a non-fresh member (external reference)
  private[chisel3] def mustClone(prevId: Long): Boolean = {
    if (this.hasBinding || this._id <= prevId) true
    else
      this match {
        case b: Bundle => b.hasExternalRef
        case _ => false
      }
  }

  override def autoSeed(name: String): this.type = {
    topBindingOpt match {
      // Ports are special in that the autoSeed will keep the first name, not the last name
      case Some(PortBinding(m)) if hasAutoSeed && Builder.currentModule.contains(m) => this
      case _                                                                        => super.autoSeed(name)
    }
  }

  // User-specified direction, local at this node only.
  // Note that the actual direction of this node can differ from child and parent specifiedDirection.
  private var _specifiedDirection:         SpecifiedDirection = SpecifiedDirection.Unspecified
  private[chisel3] def specifiedDirection: SpecifiedDirection = _specifiedDirection
  private[chisel3] def specifiedDirection_=(direction: SpecifiedDirection) = {
    _specifiedDirection = direction
  }

  /** This overwrites a relative SpecifiedDirection with an explicit one, and is used to implement
    * the compatibility layer where, at the elements, Flip is Input and unspecified is Output.
    * DO NOT USE OUTSIDE THIS PURPOSE. THIS OPERATION IS DANGEROUS!
    */
  private[chisel3] def _assignCompatibilityExplicitDirection: Unit = {
    (this, _specifiedDirection) match {
      case (_: Analog, _) => // nothing to do
      case (_, SpecifiedDirection.Unspecified)                       => _specifiedDirection = SpecifiedDirection.Output
      case (_, SpecifiedDirection.Flip)                              => _specifiedDirection = SpecifiedDirection.Input
      case (_, SpecifiedDirection.Input | SpecifiedDirection.Output) => // nothing to do
    }
  }

  // Binding stores information about this node's position in the hardware graph.
  // This information is supplemental (more than is necessary to generate FIRRTL) and is used to
  // perform checks in Chisel, where more informative error messages are possible.
  private var _bindingVar: Binding = null // using nullable var for better memory usage
  private def _binding:    Option[Binding] = Option(_bindingVar)
  // Only valid after node is bound (synthesizable), crashes otherwise
  protected[chisel3] def binding: Option[Binding] = _binding
  protected def binding_=(target: Binding) {
    if (_binding.isDefined) {
      throw RebindingException(s"Attempted reassignment of binding to $this, from: ${target}")
    }
    _bindingVar = target
  }

  private[chisel3] def hasBinding: Boolean = _binding.isDefined

  // Similar to topBindingOpt except it explicitly excludes SampleElements which are bound but not
  // hardware
  private[chisel3] final def isSynthesizable: Boolean = _binding.map {
    case ChildBinding(parent) => parent.isSynthesizable
    case _: TopBinding => true
    case (_: SampleElementBinding[_] | _: MemTypeBinding[_]) => false
  }.getOrElse(false)

  private[chisel3] def topBindingOpt: Option[TopBinding] = _binding.flatMap {
    case ChildBinding(parent) => parent.topBindingOpt
    case bindingVal: TopBinding => Some(bindingVal)
    case SampleElementBinding(parent) => parent.topBindingOpt
    case _: MemTypeBinding[_] => None
  }

  private[chisel3] def topBinding: TopBinding = topBindingOpt.get

  /** Binds this node to the hardware graph.
    * parentDirection is the direction of the parent node, or Unspecified (default) if the target
    * node is the top-level.
    * binding and direction are valid after this call completes.
    */
  private[chisel3] def bind(target: Binding, parentDirection: SpecifiedDirection = SpecifiedDirection.Unspecified): Unit

  /** Adds this `Data` to its parents _ids if it should be added */
  private[chisel3] def maybeAddToParentIds(target: Binding): Unit = {
    // ConstrainedBinding means the thing actually corresponds to a Module, no need to add to _ids otherwise
    if (target.isInstanceOf[ConstrainedBinding]) {
      _parent.foreach(_.addId(this))
    }
  }

  // Both _direction and _resolvedUserDirection are saved versions of computed variables (for
  // efficiency, avoid expensive recomputation of frequent operations).
  // Both are only valid after binding is set.

  // Direction of this node, accounting for parents (force Input / Output) and children.
  private var _directionVar: ActualDirection = null // using nullable var for better memory usage
  private def _direction:    Option[ActualDirection] = Option(_directionVar)

  private[chisel3] def direction: ActualDirection = _direction.get
  private[chisel3] def direction_=(actualDirection: ActualDirection) {
    if (_direction.isDefined) {
      throw RebindingException(s"Attempted reassignment of resolved direction to $this")
    }
    _directionVar = actualDirection
  }

  private[chisel3] def stringAccessor(chiselType: String): String = {
    topBindingOpt match {
      case None => chiselType
      // Handle DontCares specially as they are "literal-like" but not actually literals
      case Some(DontCareBinding()) => s"$chiselType(DontCare)"
      case Some(topBinding) =>
        val binding: String = _bindingToString(topBinding)
        val name = earlyName
        val mod = parentNameOpt.map(_ + ".").getOrElse("")

        s"$mod$name: $binding[$chiselType]"
    }
  }

  // User-friendly representation of the binding as a helper function for toString.
  // Provides a unhelpful fallback for literals, which should have custom rendering per
  // Data-subtype.
  private[chisel3] def _bindingToString(topBindingOpt: TopBinding): String =
    topBindingOpt match {
      case OpBinding(_, _)           => "OpResult"
      case MemoryPortBinding(_, _)   => "MemPort"
      case PortBinding(_)            => "IO"
      case RegBinding(_, _)          => "Reg"
      case WireBinding(_, _)         => "Wire"
      case DontCareBinding()         => "(DontCare)"
      case ElementLitBinding(litArg) => "(unhandled literal)"
      case BundleLitBinding(litMap)  => "(unhandled bundle literal)"
      case VecLitBinding(litMap)     => "(unhandled vec literal)"
      case _                         => ""
    }

  private[chisel3] def earlyName: String = Arg.earlyLocalName(this)

  private[chisel3] def parentNameOpt: Option[String] = this._parent.map(_.name)

  // Return ALL elements at root of this type.
  // Contasts with flatten, which returns just Bits
  // TODO: refactor away this, this is outside the scope of Data
  private[chisel3] def allElements: Seq[Element]

  private[chisel3] def badConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    throwException(s"cannot connect ${this} and ${that}")

  private[chisel3] def connect(
    that: Data
  )(
    implicit sourceInfo:   SourceInfo,
    connectCompileOptions: CompileOptions
  ): Unit = {
    if (connectCompileOptions.checkSynthesizable) {
      requireIsHardware(this, "data to be connected")
      requireIsHardware(that, "data to be connected")
      this.topBinding match {
        case _: ReadOnlyBinding => throwException(s"Cannot reassign to read-only $this")
        case _ => // fine
      }
    }
    if (connectCompileOptions.migrateMonoConnections) {
      getRecursiveFields.lazilyNoPath(this).collect {
        case d if d.direction != this.direction =>
          Builder.error(s"$this cannot be used with := because submember $d has inverse orientation; use :#= instead")
      }
      getRecursiveFields.lazilyNoPath(that).collect {
        case d if d.direction != that.direction =>
          Builder.error(s"$that cannot be used with := because submember $d has inverse orientation; use :#= instead")
      }
    }
    if (connectCompileOptions.emitStrictConnects) {

      try {
        MonoConnect.connect(sourceInfo, connectCompileOptions, this, that, Builder.referenceUserModule)
      } catch {
        case MonoConnectException(message) =>
          throwException(
            s"Connection between sink ($this) and source ($that) failed @: $message"
          )
      }
    } else {
      this.firrtlPartialConnect(that)
    }
  }
  private[chisel3] def bulkConnect(
    that: Data
  )(
    implicit sourceInfo:   SourceInfo,
    connectCompileOptions: CompileOptions
  ): Unit = {
    if (connectCompileOptions.checkSynthesizable) {
      requireIsHardware(this, s"data to be bulk-connected")
      requireIsHardware(that, s"data to be bulk-connected")
      (this.topBinding, that.topBinding) match {
        case (_: ReadOnlyBinding, _: ReadOnlyBinding) => throwException(s"Both $this and $that are read-only")
        // DontCare cannot be a sink (LHS)
        case (_: DontCareBinding, _) => throw BiConnect.DontCareCantBeSink
        case _ => // fine
      }
    }
    if (connectCompileOptions.emitStrictConnects) {
      try {
        BiConnect.connect(sourceInfo, connectCompileOptions, this, that, Builder.referenceUserModule)
      } catch {
        case BiConnectException(message) =>
          throwException(
            s"Connection between left ($this) and source ($that) failed @$message"
          )
      }
    } else {
      if (connectCompileOptions.migrateBulkConnections)
        Builder.error(s"Cannot use <> in an `import Chisel._` file; use :<>= instead")
      this.firrtlPartialConnect(that)
    }
  }

  /** Whether this Data has the same model ("data type") as that Data.
    * Data subtypes should overload this with checks against their own type.
    */
  private[chisel3] def typeEquivalent(that: Data): Boolean

  private[chisel3] def requireVisible(): Unit = {
    val mod = topBindingOpt.flatMap(_.location)
    topBindingOpt match {
      case Some(tb: TopBinding) if (mod == Builder.currentModule) =>
      case Some(pb: PortBinding)
          if (mod.flatMap(Builder.retrieveParent(_, Builder.currentModule.get)) == Builder.currentModule) =>
      case Some(_: UnconstrainedBinding) =>
      case _ =>
        throwException(s"operand '$this' is not visible from the current module")
    }
    if (!MonoConnect.checkWhenVisibility(this)) {
      throwException(s"operand has escaped the scope of the when in which it was constructed")
    }
  }

  // Internal API: returns a ref that can be assigned to, if consistent with the binding
  private[chisel3] def lref: Node = {
    requireIsHardware(this)
    requireVisible()
    topBindingOpt match {
      case Some(binding: ReadOnlyBinding) =>
        throwException(s"internal error: attempted to generate LHS ref to ReadOnlyBinding $binding")
      case Some(ViewBinding(target)) => reify(target).lref
      case Some(binding: TopBinding) => Node(this)
      case opt => throwException(s"internal error: unknown binding $opt in generating LHS ref")
    }
  }

  // Internal API: returns a ref, if bound
  private[chisel3] final def ref: Arg = {
    def materializeWire(): Arg = {
      if (!Builder.currentModule.isDefined) throwException(s"internal error: cannot materialize ref for $this")
      implicit val compileOptions = ExplicitCompileOptions.Strict
      implicit val sourceInfo = UnlocatableSourceInfo
      WireDefault(this).ref
    }
    requireIsHardware(this)
    topBindingOpt match {
      // DataView
      case Some(ViewBinding(target)) => reify(target).ref
      case Some(AggregateViewBinding(viewMap)) =>
        viewMap.get(this) match {
          case None => materializeWire() // FIXME FIRRTL doesn't have Aggregate Init expressions
          // This should not be possible because Element does the lookup in .topBindingOpt
          case x: Some[_] => throwException(s"Internal Error: In .ref for $this got '$topBindingOpt' and '$x'")
        }
      // Literals
      case Some(ElementLitBinding(litArg)) => litArg
      case Some(BundleLitBinding(litMap)) =>
        litMap.get(this) match {
          case Some(litArg) => litArg
          case _            => materializeWire() // FIXME FIRRTL doesn't have Bundle literal expressions
        }
      case Some(VecLitBinding(litMap)) =>
        litMap.get(this) match {
          case Some(litArg) => litArg
          case _            => materializeWire() // FIXME FIRRTL doesn't have Vec literal expressions
        }
      case Some(DontCareBinding()) =>
        materializeWire() // FIXME FIRRTL doesn't have a DontCare expression so materialize a Wire
      // Non-literals
      case Some(binding: TopBinding) =>
        if (Builder.currentModule.isDefined) {
          // This is allowed (among other cases) for evaluating args of Printf / Assert / Printable, which are
          // partially resolved *after* elaboration completes. If this is resolved, the check should be unconditional.
          requireVisible()
        }
        Node(this)
      case opt => throwException(s"internal error: unknown binding $opt in generating LHS ref")
    }
  }

  // Recursively set the parent of the start Data and any children (eg. in an Aggregate)
  private[chisel3] def setAllParents(parent: Option[BaseModule]): Unit = {
    def rec(data: Data): Unit = {
      data._parent = parent
      data match {
        case _:   Element =>
        case agg: Aggregate =>
          agg.elementsIterator.foreach(rec)
      }
    }
    rec(this)
  }

  private[chisel3] def width: Width
  private[chisel3] def firrtlConnect(that:        Data)(implicit sourceInfo: SourceInfo): Unit
  private[chisel3] def firrtlPartialConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit

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
    val clone = this.cloneType // get a fresh object, without bindings
    // Only the top-level direction needs to be fixed up, cloneType should do the rest
    clone.specifiedDirection = specifiedDirection
    clone
  }

  /** The "strong connect" operator.
    *
    * For chisel3._, this operator is mono-directioned; all sub-elements of `this` will be driven by sub-elements of `that`.
    *  - Equivalent to `this :#= that`
    *
    * For Chisel._, this operator connections bi-directionally via emitting the FIRRTL.<=
    *  - Equivalent to `this :<>= that`
    *
    * @param that the Data to connect from
    * @group connection
    */
  final def :=(that: => Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit = {
    prefix(this) {
      this.connect(that)(sourceInfo, connectionCompileOptions)
    }
  }

  /** The "bulk connect operator", assigning elements in this Vec from elements in a Vec.
    *
    * For chisel3._, uses the [[BiConnect]] algorithm; sub-elements of `that` may end up driving sub-elements of `this`
    *  - Complicated semantics, hard to write quickly, will likely be deprecated in the future
    *
    * For Chisel._, emits the FIRRTL.<- operator
    *  - Equivalent to `this :<>= that` without the restrictions that bundle field names and vector sizes must match
    *
    * @param that the Data to connect from
    * @group connection
    */
  final def <>(that: => Data)(implicit sourceInfo: SourceInfo, connectionCompileOptions: CompileOptions): Unit = {
    prefix(this) {
      this.bulkConnect(that)(sourceInfo, connectionCompileOptions)
    }
  }

  def isLit: Boolean = litOption.isDefined

  /**
    * If this is a literal that is representable as bits, returns the value as a BigInt.
    * If not a literal, or not representable as bits (for example, is or contains Analog), returns None.
    */
  def litOption: Option[BigInt]

  /**
    * Returns the literal value if this is a literal that is representable as bits, otherwise crashes.
    */
  def litValue: BigInt = litOption.get

  /** Returns the width, in bits, if currently known. */
  final def getWidth: Int =
    if (isWidthKnown) width.get else throwException(s"Width of $this is unknown!")

  /** Returns whether the width is currently known. */
  final def isWidthKnown: Boolean = width.known

  /** Returns Some(width) if the width is known, else None. */
  final def widthOption: Option[Int] = if (isWidthKnown) Some(getWidth) else None

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
    thatCloned.connectFromBits(this.asUInt)
    thatCloned
  }

  /** Assigns this node from Bits type. Internal implementation for asTypeOf.
    */
  private[chisel3] def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Unit

  /** Reinterpret cast to UInt.
    *
    * @note value not guaranteed to be preserved: for example, a SInt of width
    * 3 and value -1 (0b111) would become an UInt with value 7
    * @note Aggregates are recursively packed with the first element appearing
    * in the least-significant bits of the result.
    */
  final def asUInt: UInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt

  /** Default pretty printing */
  def toPrintable: Printable
}

object Data {
  // Needed for the `implicit def toConnectableDefault`
  import scala.language.implicitConversions

  /** Provides :<=, :>=, :<>=, and :#= between consumer and producer of the same T <: Data */
  implicit class ConnectableDefault[T <: Data](consumer: T) extends connectable.ConnectableOperators[T](consumer)

  /** Provides :<>=, :<=, :>=, and :#= between a (consumer: Vec) and (producer: Seq) */
  implicit class ConnectableVecDefault[T <: Data](consumer: Vec[T])
      extends connectable.ConnectableVecOperators[T](consumer)

  /** Can implicitly convert a Data to a Connectable
    *
    * Originally this was done with an implicit class, but all functions we want to
    *  add to Data we also want on Connectable, so an implicit conversion makes the most sense
    *  so the ScalaDoc can be shared.
    */
  implicit def toConnectableDefault[T <: Data](d: T): Connectable[T] = Connectable.apply(d)

  /** Typeclass implementation of HasMatchingZipOfChildren for Data
    *
    * The canonical API to iterate through two Chisel types or components, where
    *   matching children are provided together, while non-matching members are provided
    *   separately
    *
    * Only zips immediate children (vs members, which are all children/grandchildren etc.)
    */
  implicit private[chisel3] val DataMatchingZipOfChildren = new DataMirror.HasMatchingZipOfChildren[Data] {

    implicit class VecOptOps(vOpt: Option[Vec[Data]]) {
      // Like .get, but its already defined on Option
      def grab(i: Int): Option[Data] = vOpt.flatMap { _.lift(i) }
      def size = vOpt.map(_.size).getOrElse(0)
    }
    implicit class RecordOptGet(rOpt: Option[Record]) {
      // Like .get, but its already defined on Option
      def grab(k: String): Option[Data] = rOpt.flatMap { _.elements.get(k) }
      def keys: Iterable[String] = rOpt.map { r => r.elements.map(_._1) }.getOrElse(Seq.empty[String])
    }
    //TODO(azidar): Rewrite this to be more clear, probably not the cleanest way to express this
    private def isDifferent(l: Option[Data], r: Option[Data]): Boolean =
      l.nonEmpty && r.nonEmpty && !isRecord(l, r) && !isVec(l, r) && !isElement(l, r)
    private def isRecord(l: Option[Data], r: Option[Data]): Boolean =
      l.orElse(r).map { _.isInstanceOf[Record] }.getOrElse(false)
    private def isVec(l: Option[Data], r: Option[Data]): Boolean =
      l.orElse(r).map { _.isInstanceOf[Vec[_]] }.getOrElse(false)
    private def isElement(l: Option[Data], r: Option[Data]): Boolean =
      l.orElse(r).map { _.isInstanceOf[Element] }.getOrElse(false)

    /** Zips matching children of `left` and `right`; returns Nil if both are empty
      *
      * The canonical API to iterate through two Chisel types or components, where
      * matching children are provided together, while non-matching members are provided
      * separately
      *
      * Only zips immediate children (vs members, which are all children/grandchildren etc.)
      *
      * Returns Nil if both are different types
      */
    def matchingZipOfChildren(left: Option[Data], right: Option[Data]): Seq[(Option[Data], Option[Data])] =
      (left, right) match {
        case (None, None)                            => Nil
        case (lOpt, rOpt) if isDifferent(lOpt, rOpt) => Nil
        case (lOpt: Option[Vec[Data] @unchecked], rOpt: Option[Vec[Data] @unchecked]) if isVec(lOpt, rOpt) =>
          (0 until (lOpt.size.max(rOpt.size))).map { i => (lOpt.grab(i), rOpt.grab(i)) }
        case (lOpt: Option[Record @unchecked], rOpt: Option[Record @unchecked]) if isRecord(lOpt, rOpt) =>
          (lOpt.keys ++ rOpt.keys).toList.distinct.map { k => (lOpt.grab(k), rOpt.grab(k)) }
        case (lOpt, rOpt) if isElement(lOpt, rOpt) => Nil
      }
  }

  /**
    * Provides generic, recursive equality for [[Bundle]] and [[Vec]] hardware. This avoids the
    * need to use workarounds such as `bundle1.asUInt === bundle2.asUInt` by allowing users
    * to instead write `bundle1 === bundle2`.
    *
    * Static type safety of this comparison is guaranteed at compile time as the extension
    * method requires the same parameterized type for both the left-hand and right-hand
    * sides. It is, however, possible to get around this type safety using `Bundle` subtypes
    * that can differ during runtime (e.g. through a generator). These cases are
    * subsequently raised as elaboration errors.
    *
    * @param lhs The [[Data]] hardware on the left-hand side of the equality
    */
  implicit class DataEquality[T <: Data](lhs: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {

    /** Dynamic recursive equality operator for generic [[Data]]
      *
      * @param rhs a hardware [[Data]] to compare `lhs` to
      * @return a hardware [[Bool]] asserted if `lhs` is equal to `rhs`
      * @throws ChiselException when `lhs` and `rhs` are different types during elaboration time
      */
    def ===(rhs: T): Bool = {
      (lhs, rhs) match {
        case (thiz: UInt, that: UInt) => thiz === that
        case (thiz: SInt, that: SInt) => thiz === that
        case (thiz: AsyncReset, that: AsyncReset) => thiz.asBool === that.asBool
        case (thiz: Reset, that: Reset) => thiz === that
        case (thiz: Interval, that: Interval) => thiz === that
        case (thiz: FixedPoint, that: FixedPoint) => thiz === that
        case (thiz: EnumType, that: EnumType) => thiz === that
        case (thiz: Clock, that: Clock) => thiz.asUInt === that.asUInt
        case (thiz: Vec[_], that: Vec[_]) =>
          if (thiz.length != that.length) {
            throwException(s"Cannot compare Vecs $thiz and $that: Vec sizes differ")
          } else {
            thiz.elementsIterator
              .zip(that.elementsIterator)
              .map { case (thisData, thatData) => thisData === thatData }
              .reduce(_ && _)
          }
        case (thiz: Record, that: Record) =>
          if (thiz.elements.size != that.elements.size) {
            throwException(s"Cannot compare Bundles $thiz and $that: Bundle types differ")
          } else {
            thiz.elements.map {
              case (thisName, thisData) =>
                if (!that.elements.contains(thisName))
                  throwException(
                    s"Cannot compare Bundles $thiz and $that: field $thisName (from $thiz) was not found in $that"
                  )

                val thatData = that.elements(thisName)

                try {
                  thisData === thatData
                } catch {
                  case e: ChiselException =>
                    throwException(
                      s"Cannot compare field $thisName in Bundles $thiz and $that: ${e.getMessage.split(": ").last}"
                    )
                }
            }
              .reduce(_ && _)
          }
        // This should be matching to (DontCare, DontCare) but the compiler wasn't happy with that
        case (_: DontCare.type, _: DontCare.type) => true.B

        case (thiz: Analog, that: Analog) =>
          throwException(s"Cannot compare Analog values $thiz and $that: Equality isn't defined for Analog values")
        // Runtime types are different
        case (thiz, that) => throwException(s"Cannot compare $thiz and $that: Runtime types differ")
      }
    }
  }
}

trait WireFactory {

  /** Construct a [[Wire]] from a type template
    * @param t The template from which to construct this wire
    */
  def apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    val prevId = Builder.idGen.value
    val t = source // evaluate once (passed by name)
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(t, "wire type")
    }
    val x = if (!t.mustClone(prevId)) t else t.cloneTypeFull

    // Bind each element of x to being a Wire
    x.bind(WireBinding(Builder.forcedUserModule, Builder.currentWhen))

    pushCommand(DefWire(sourceInfo, x))
    if (!compileOptions.explicitInvalidate || Builder.currentModule.get.isInstanceOf[ImplicitInvalidate]) {
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

  private def applyImpl[T <: Data](
    t:    T,
    init: Data
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
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
  def apply[T <: Data](
    t:    T,
    init: DontCare.type
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
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
final case object DontCare extends Element with connectable.ConnectableDocs {
  // This object should be initialized before we execute any user code that refers to it,
  //  otherwise this "Chisel" object will end up on the UserModule's id list.
  // We make it private to chisel3 so it has to be accessed through the package object.

  private[chisel3] override val width: Width = UnknownWidth()

  bind(DontCareBinding(), SpecifiedDirection.Output)
  override def cloneType: this.type = DontCare

  override def toString: String = "DontCare()"

  override def litOption: Option[BigInt] = None

  def toPrintable: Printable = PString("DONTCARE")

  private[chisel3] def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Unit = {
    Builder.error("connectFromBits: DontCare cannot be a connection sink (LHS)")
  }

  def do_asUInt(implicit sourceInfo: chisel3.internal.sourceinfo.SourceInfo, compileOptions: CompileOptions): UInt = {
    Builder.error("DontCare does not have a UInt representation")
    0.U
  }
  // DontCare's only match themselves.
  private[chisel3] def typeEquivalent(that: Data): Boolean = that == DontCare

  /** $colonGreaterEq
    *
    * @group connection
    * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
    */
  final def :>=[T <: Data](producer: => T)(implicit sourceInfo: SourceInfo): Unit =
    this.asInstanceOf[Data] :>= producer.asInstanceOf[Data]
}
