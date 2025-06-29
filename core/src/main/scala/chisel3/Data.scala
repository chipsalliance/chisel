// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.dataview.reify

import chisel3.experimental.{requireIsChiselType, requireIsHardware, Analog, BaseModule}
import chisel3.experimental.{prefix, SourceInfo, UnlocatableSourceInfo}
import chisel3.experimental.dataview.{reifyIdentityView, reifySingleTarget, DataViewable}
import chisel3.internal.Builder.pushCommand
import chisel3.internal._
import chisel3.internal.binding._
import chisel3.internal.firrtl.ir._
import chisel3.properties.Property
import chisel3.reflect.DataMirror
import chisel3.util.simpleClassName

import scala.reflect.ClassTag
import scala.util.Try
import scala.util.control.NonFatal

/** User-specified directions.
  */
sealed abstract class SpecifiedDirection(private[chisel3] val value: Byte)
object SpecifiedDirection {

  /** Default user direction, also meaning 'not-flipped'
    */
  case object Unspecified extends SpecifiedDirection(0)

  /** Node and its children are forced as output
    */
  case object Output extends SpecifiedDirection(1)

  /** Node and its children are forced as inputs
    */
  case object Input extends SpecifiedDirection(2)

  /** Mainly for containers, children are flipped.
    */
  case object Flip extends SpecifiedDirection(3)

  private[chisel3] def fromByte(b: Byte): SpecifiedDirection = b match {
    case Unspecified.value => Unspecified
    case Output.value      => Output
    case Input.value       => Input
    case Flip.value        => Flip
    case _                 => throw new RuntimeException(s"Unexpected SpecifiedDirection value $b")
  }

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
  )(dir: T => SpecifiedDirection): T = {
    val prevId = Builder.idGen.value
    val data = source // evaluate source once (passed by name)
    requireIsChiselType(data)
    val out = if (!data.mustClone(prevId)) data else data.cloneTypeFull.asInstanceOf[T]
    out.specifiedDirection = dir(data) // Must use original data, specified direction of clone is cleared
    out
  }

}

/** Resolved directions for both leaf and container nodes, only visible after
  * a node is bound (since higher-level specifications like Input and Output
  * can override directions).
  */
sealed abstract class ActualDirection(private[chisel3] val value: Byte)

object ActualDirection {

  // 0 is reserved for unset, no case object added because that would be an unnecessary API breakage.
  private[chisel3] val Unset: Byte = 0

  /** The object does not exist / is empty and hence has no direction
    */
  case object Empty extends ActualDirection(1)

  /** Undirectioned, struct-like
    */
  case object Unspecified extends ActualDirection(2)

  /** Output element, or container with all outputs (even if forced)
    */
  case object Output extends ActualDirection(3)

  /** Input element, or container with all inputs (even if forced)
    */
  case object Input extends ActualDirection(4)

  // BidirectionalDirection is effectively an extension of ActualDirection, see its use in Bidirectional below.
  // Thus, the numbering here is a continuation of the numbering in ActualDirection.
  // Bidirectional.Default and Bidirectional.Flipped wrap these objects.
  sealed abstract class BidirectionalDirection(private[chisel3] val value: Byte)
  case object Default extends BidirectionalDirection(5)
  case object Flipped extends BidirectionalDirection(6)

  // This constructor has 2 arguments (which need to be in sync) only to distinguish it from the other constructor
  // Once that one is removed, delete _value
  case class Bidirectional private[chisel3] (dir: BidirectionalDirection, _value: Byte)
      extends ActualDirection(_value) {
    @deprecated("Use companion object factory apply method", "Chisel 6.5")
    def this(dir: BidirectionalDirection) = this(dir, dir.value)
    private[chisel3] def copy(dir: BidirectionalDirection = this.dir, _value: Byte = this._value) =
      new Bidirectional(dir, _value)
  }
  object Bidirectional {
    val Default = new Bidirectional(ActualDirection.Default, ActualDirection.Default.value)
    val Flipped = new Bidirectional(ActualDirection.Flipped, ActualDirection.Flipped.value)
    def apply(dir: BidirectionalDirection): ActualDirection = dir match {
      case ActualDirection.Default => Default
      case ActualDirection.Flipped => Flipped
    }
    @deprecated("Match on Bidirectional.Default and Bidirectional.Flipped directly instead", "Chisel 6.5")
    def unapply(dir:                Bidirectional): Option[BidirectionalDirection] = Some(dir.dir)
    private[chisel3] def apply(dir: BidirectionalDirection, _value: Byte) = new Bidirectional(dir, _value)
  }

  private[chisel3] def fromByte(b: Byte): ActualDirection = b match {
    case Empty.value                 => Empty
    case Unspecified.value           => Unspecified
    case Output.value                => Output
    case Input.value                 => Input
    case Bidirectional.Default.value => Bidirectional.Default
    case Bidirectional.Flipped.value => Bidirectional.Flipped
    case _                           => throwException(s"Unexpected ActualDirection value $b")
  }

  /** Converts a `SpecifiedDirection` to an `ActualDirection`
    *
    * Implements the Chisel convention that Flip is Input and unspecified is Output.
    */
  def fromSpecified(direction: SpecifiedDirection): ActualDirection = direction match {
    case SpecifiedDirection.Output | SpecifiedDirection.Unspecified => ActualDirection.Output
    case SpecifiedDirection.Input | SpecifiedDirection.Flip         => ActualDirection.Input
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
    implicit sourceInfo: SourceInfo
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
        val mismatch =
          elt.findFirstTypeMismatch(filteredElts.head, strictTypes = true, strictWidths = true, strictProbeInfo = true)
        require(
          mismatch.isEmpty,
          s"can't create $createdType with non-equivalent types _${mismatch.get}"
        )
      }
      filteredElts.head.cloneTypeFull
    }
  }
}

// Returns pairs of all fields, element-level and containers, in a Record and their path names
private[chisel3] object getRecursiveFields {
  def noPath(data:       Data): Seq[Data] = lazilyNoPath(data).toVector
  def lazilyNoPath(data: Data): Iterable[Data] = DataMirror.collectMembers(data) { case x => x }

  def apply(data: Data, path: String): Seq[(Data, String)] = lazily(data, path).toVector
  def lazily(data: Data, path: String): Iterable[(Data, String)] = DataMirror.collectMembersAndPaths(data, path) {
    case x => x
  }
}

// Returns pairs of corresponding fields between two Records of the same type
// TODO it seems wrong that Elements are checked for typeEquivalence in Bundle and Vec lit creation
private[chisel3] object getMatchedFields {
  def apply(x: Data, y: Data): Seq[(Data, Data)] = (x, y) match {
    case (x: Element, y: Element) =>
      x.requireTypeEquivalent(y)
      Seq(x -> y)
    case (_, _) if DataMirror.hasProbeTypeModifier(x) || DataMirror.hasProbeTypeModifier(y) => {
      x.requireTypeEquivalent(y)
      Seq(x -> y)
    }
    case (x: Record, y: Record) =>
      (x._elements
        .zip(y._elements))
        .map { case ((xName, xElt), (yName, yElt)) =>
          require(
            xName == yName,
            s"$xName != $yName, ${x._elements}, ${y._elements}, $x, $y"
          ) // assume fields returned in same, deterministic order
          getMatchedFields(xElt, yElt)
        }
        .fold(Seq(x -> y)) {
          _ ++ _
        }
    case (x: Vec[_], y: Vec[_]) =>
      (x.elementsIterator
        .zip(y.elementsIterator))
        .map { case (xElt, yElt) =>
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
  def apply[T <: Data](source: => T): T = {
    SpecifiedDirection.specifiedDirection(source)(_ => SpecifiedDirection.Input)
  }
}
object Output {
  def apply[T <: Data](source: => T): T = {
    SpecifiedDirection.specifiedDirection(source)(_ => SpecifiedDirection.Output)
  }
}

object Flipped {
  def apply[T <: Data](source: => T): T = {
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
abstract class Data extends HasId with NamedComponent with DataIntf {
  import Data.ProbeInfo

  // This is a bad API that punches through object boundaries.
  private[chisel3] def flatten: IndexedSeq[Element] = {
    this match {
      case elt: Aggregate => elt.elementsIterator.toIndexedSeq.flatMap { _.flatten }
      case elt: Element   => IndexedSeq(elt)
      case elt => throwException(s"Cannot flatten type ${elt.getClass}")
    }
  }

  // Whether this node'e element(s) possess a specified flipped direction, ignoring coercion via Input/Output
  private[chisel3] def containsAFlipped: Boolean = false

  // Must clone a Data if any of the following are true:
  // * It has a binding
  // * Its id is older than prevId (not "freshly created")
  // * It is a Bundle or Record that contains a member older than prevId
  private[chisel3] def mustClone(prevId: Long): Boolean = {
    this.hasBinding || this._minId <= prevId
  }

  /** The minimum (aka "oldest") id that is part of this Data
    *
    * @note This is usually just _id except for some Records and Bundles
    */
  private[chisel3] def _minId: Long = this._id

  override def autoSeed(name: String): this.type = {
    topBindingOpt match {
      // Ports are special in that the autoSeed will keep the first name, not the last name
      case Some(PortBinding(m)) if hasSeed && Builder.currentModule.contains(m) => this
      case _                                                                    => super.autoSeed(name)
    }
  }

  // probeInfo only exists if this is a probe type
  private var _probeInfoVar:      ProbeInfo = null
  private[chisel3] def probeInfo: Option[ProbeInfo] = Option(_probeInfoVar)
  private[chisel3] def probeInfo_=(probeInfo: Option[ProbeInfo]) = _probeInfoVar = probeInfo.getOrElse(null)

  // If this Data is constant, it must hold a constant value
  private var _isConst:         Boolean = false
  private[chisel3] def isConst: Boolean = _isConst
  private[chisel3] def isConst_=(isConst: Boolean) = _isConst = isConst

  // Both _direction and _resolvedUserDirection are saved versions of computed variables (for
  // efficiency, avoid expensive recomputation of frequent operations).
  // Both are only valid after binding is set.

  // User-specified direction, local at this node only.
  // Note that the actual direction of this node can differ from child and parent specifiedDirection.
  private var _specifiedDirection:         Byte = SpecifiedDirection.Unspecified.value
  private[chisel3] def specifiedDirection: SpecifiedDirection = SpecifiedDirection.fromByte(_specifiedDirection)
  private[chisel3] def specifiedDirection_=(direction: SpecifiedDirection) = {
    _specifiedDirection = direction.value
  }

  // Direction of this node, accounting for parents (force Input / Output) and children.
  private var _directionVar: Byte = ActualDirection.Unset
  private def _direction: Option[ActualDirection] =
    Option.when(_directionVar != ActualDirection.Unset)(ActualDirection.fromByte(_directionVar))

  private[chisel3] def direction: ActualDirection = _direction.get
  private[chisel3] def direction_=(actualDirection: ActualDirection): Unit = {
    if (_direction.isDefined) {
      throw RebindingException(s"Attempted reassignment of resolved direction to $this")
    }
    _directionVar = actualDirection.value
  }

  /** This overwrites a relative SpecifiedDirection with an explicit one, and is used to implement
    * the compatibility layer where, at the elements, Flip is Input and unspecified is Output.
    * DO NOT USE OUTSIDE THIS PURPOSE. THIS OPERATION IS DANGEROUS!
    */
  private[chisel3] def _assignCompatibilityExplicitDirection: Unit = {
    (this, specifiedDirection) match {
      case (_: Analog, _)                                            => // nothing to do
      case (_, SpecifiedDirection.Unspecified)                       => specifiedDirection = SpecifiedDirection.Output
      case (_, SpecifiedDirection.Flip)                              => specifiedDirection = SpecifiedDirection.Input
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
  protected def binding_=(target: Binding): Unit = {
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
    case (_: SampleElementBinding[_] | _: MemTypeBinding[_] | _: FirrtlMemTypeBinding) => false
  }.getOrElse(false)

  private[chisel3] def topBindingOpt: Option[TopBinding] = _binding.flatMap {
    case ChildBinding(parent) => parent.topBindingOpt
    case bindingVal: TopBinding => Some(bindingVal)
    case SampleElementBinding(parent)                     => parent.topBindingOpt
    case (_: MemTypeBinding[_] | _: FirrtlMemTypeBinding) => None
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
    target match {
      case c: SecretPortBinding  => // secret ports are handled differently, parent's don't need to know about that
      case c: ConstrainedBinding => _parent.foreach(_.addId(this))
      case _ =>
    }
  }

  // Specializes the .toString method of a [[Data]] for conditions such as
  //  DataView, Probe modifiers, a DontCare, and whether it is bound or a pure chisel type
  private[chisel3] def stringAccessor(chiselType: String): String = {
    // Add probe and layer color (if they exist) to the returned String
    val chiselTypeWithModifier =
      probeInfo match {
        case None => chiselType
        case Some(ProbeInfo(writeable, layer)) =>
          val layerString = layer.map(x => s"[${x.fullName}]").getOrElse("")
          (if (writeable) "RWProbe" else "Probe") + s"$layerString<$chiselType>"
      }
    // Trace views to give better error messages
    // Reifying involves checking against ViewParent which requires being in a Builder context
    // Since we're just printing a String, suppress such errors and use this object
    val thiz = Try(reifySingleTarget(this)).toOption.flatten.getOrElse(this)
    thiz.topBindingOpt match {
      case None => chiselTypeWithModifier
      // Handle DontCares specially as they are "literal-like" but not actually literals
      case Some(DontCareBinding()) => s"$chiselType(DontCare)"
      case Some(topBinding) =>
        val binding: String = thiz._bindingToString(topBinding)
        val name = thiz.earlyName
        val mod = thiz.parentNameOpt.map(_ + ".").getOrElse("")

        s"$mod$name: $binding[$chiselTypeWithModifier]"
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
      case SecretPortBinding(_)      => "IO"
      case RegBinding(_, _)          => "Reg"
      case WireBinding(_, _)         => "Wire"
      case DontCareBinding()         => "(DontCare)"
      case ElementLitBinding(litArg) => "(unhandled literal)"
      case BundleLitBinding(litMap)  => "(unhandled bundle literal)"
      case VecLitBinding(litMap)     => "(unhandled vec literal)"
      case DynamicIndexBinding(vec)  => _bindingToString(vec.topBinding)
      case _                         => ""
    }

  private[chisel3] def earlyName: String = Arg.earlyLocalName(this)

  // Only used in error messages, this is not allowed to fail
  private[chisel3] def parentNameOpt: Option[String] = try {
    this._parent.map(_.name)
  } catch {
    case NonFatal(_) => Some("<unknown>")
  }

  /** Useful information for recoverable errors that will allow the error to deduplicate */
  private[chisel3] def _localErrorContext: String = {
    if (this.binding.exists(_.isInstanceOf[ChildBinding])) {
      val n = Arg.earlyLocalName(this, includeRoot = false)
      s"Field '$n' of type ${this.typeName}"
    } else {
      this.typeName
    }
  }

  // Return ALL elements at root of this type.
  // Contasts with flatten, which returns just Bits
  // TODO: refactor away this, this is outside the scope of Data
  private[chisel3] def allElements: Seq[Element]

  private[chisel3] def badConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    throwException(s"cannot connect ${this} and ${that}")

  private[chisel3] def connect(
    that: Data
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    requireIsHardware(this, "data to be connected")
    requireIsHardware(that, "data to be connected")
    this.topBinding match {
      case _: ReadOnlyBinding => throwException(s"Cannot reassign to read-only $this")
      case _ => // fine
    }

    try {
      MonoConnect.connect(sourceInfo, this, that, Builder.referenceUserContainer)
    } catch {
      case MonoConnectException(message) =>
        throwException(
          s"Connection between sink ($this) and source ($that) failed @: $message"
        )
    }
  }
  private[chisel3] def bulkConnect(
    that: Data
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    requireIsHardware(this, s"data to be bulk-connected")
    requireIsHardware(that, s"data to be bulk-connected")
    (this.topBinding, that.topBinding) match {
      case (_: ReadOnlyBinding, _: ReadOnlyBinding) => throwException(s"Both $this and $that are read-only")
      // DontCare cannot be a sink (LHS)
      case (_: DontCareBinding, _) => throw BiConnect.DontCareCantBeSink
      case _                       => // fine
    }
    try {
      BiConnect.connect(sourceInfo, this, that, Builder.referenceUserModule)
    } catch {
      case BiConnectException(message) =>
        throwException(
          s"Connection between left ($this) and source ($that) failed @$message"
        )
    }
  }

  /** Whether this Data has the same model ("data type") as that Data.
    * Data subtypes should overload this with checks against their own type.
    * @param that the Data to check for type equivalence against.
    * @param strictProbeInfo whether probe info (including its RW-ness and Color) must match
    */
  private[chisel3] final def typeEquivalent(
    that:            Data,
    strictProbeInfo: Boolean = true
  ): Boolean =
    findFirstTypeMismatch(that, strictTypes = true, strictWidths = true, strictProbeInfo = strictProbeInfo).isEmpty

  /** Find and report any type mismatches
    *
    * @param that Data being compared to this
    * @param strictTypes Does class of Bundles or Records need to match? Inverse of "structural".
    * @param strictWidths do widths need to match?
    * @param strictProbeInfo does probe info need to match (includes RW and Color)
    * @return None if types are equivalent, Some String reporting the first mismatch if not
    */
  private[chisel3] final def findFirstTypeMismatch(
    that:            Data,
    strictTypes:     Boolean,
    strictWidths:    Boolean,
    strictProbeInfo: Boolean
  ): Option[String] = {

    def checkProbeInfo(left: Data, right: Data): Option[String] =
      Option.when(strictProbeInfo && (left.probeInfo != right.probeInfo)) {
        def probeInfoStr(info: Option[ProbeInfo]) = info.map { info =>
          s"Some(writeable=${info.writable}, color=${info.color})"
        }.getOrElse("None")
        s": Left ($left with probeInfo: ${probeInfoStr(left.probeInfo)}) and Right ($right with probeInfo: ${probeInfoStr(right.probeInfo)}) have different probeInfo."
      }

    def rec(left: Data, right: Data): Option[String] =
      checkProbeInfo(left, right).orElse {
        (left, right) match {
          // Careful, EnumTypes are Element and if we don't implement this, then they are all always equal
          case (e1: EnumType, e2: EnumType) =>
            // TODO, should we implement a form of structural equality for enums?
            if (e1.factory == e2.factory) None
            else Some(s": Left ($e1) and Right ($e2) have different types.")
          // Properties should be considered equal when getPropertyType is equal, not when getClass is equal.
          case (p1: Property[_], p2: Property[_]) =>
            if (p1.getPropertyType != p2.getPropertyType) {
              Some(s": Left ($p1) and Right ($p2) have different types")
            } else {
              None
            }
          case (e1: Element, e2: Element) if e1.getClass == e2.getClass =>
            if (strictWidths && e1.width != e2.width) {
              Some(s": Left ($e1) and Right ($e2) have different widths.")
            } else {
              None
            }
          case (r1: Record, r2: Record) if !strictTypes || r1.getClass == r2.getClass =>
            val (larger, smaller, msg) =
              if (r1._elements.size >= r2._elements.size) (r1, r2, "Left") else (r2, r1, "Right")
            larger._elements.flatMap { case (name, data) =>
              val recurse = smaller._elements.get(name) match {
                case None        => Some(s": Dangling field on $msg")
                case Some(data2) => rec(data, data2)
              }
              recurse.map("." + name + _)
            }.headOption
          case (v1: Vec[_], v2: Vec[_]) =>
            if (v1.size != v2.size) {
              Some(s": Left (size ${v1.size}) and Right (size ${v2.size}) have different lengths.")
            } else {
              val recurse = rec(v1.sample_element, v2.sample_element)
              recurse.map("[_]" + _)
            }
          case _ => Some(s": Left ($left) and Right ($right) have different types.")
        }
      }

    rec(this, that)
  }

  /** Require that two things are type equivalent, and if they are not, print a helpful error message as
    * to why not.
    *
    * @param that the Data to compare to for type equivalence
    * @param message if they are not type equivalent, contextual message to add to the exception thrown
    */
  private[chisel3] def requireTypeEquivalent(that: Data, message: String = ""): Unit = {
    require(
      this.typeEquivalent(that), {
        val reason = this
          .findFirstTypeMismatch(that, strictTypes = true, strictWidths = true, strictProbeInfo = true)
          .map(s => s"\nbecause $s")
          .getOrElse("")
        s"$message$this is not typeEquivalent to $that$reason"
      }
    )
  }

  private[chisel3] def isVisible: Boolean = isVisibleFromModule && visibleFromBlock.isEmpty
  private[chisel3] def isVisibleFromModule: Boolean = {
    val topBindingOpt = this.topBindingOpt // Only call the function once
    val mod = topBindingOpt.flatMap(_.location)
    topBindingOpt match {
      case Some(tb: TopBinding) if (mod == Builder.currentModule) => true
      case Some(pb: PortBinding)
          if mod.flatMap(Builder.retrieveParent(_, Builder.currentModule.get)) == Builder.currentModule =>
        true
      case Some(ViewBinding(target, _))           => target.isVisibleFromModule
      case Some(AggregateViewBinding(mapping, _)) => mapping.values.forall(_.isVisibleFromModule)
      case Some(DynamicIndexBinding(vec)) => vec.isVisibleFromModule // Use underlying Vec visibility for dynamic index
      case Some(pb: SecretPortBinding)    => true // Ignore secret to not require visibility
      case Some(_: UnconstrainedBinding)  => true
      case _                              => false
    }
  }
  private[chisel3] def visibleFromBlock: Option[SourceInfo] = MonoConnect.checkBlockVisibility(this)
  private[chisel3] def requireVisible()(implicit info: SourceInfo): Unit = {
    this.checkVisible.foreach(err => Builder.error(err))
  }
  // Some is an error message, None means no error
  private[chisel3] def checkVisible(implicit info: SourceInfo): Option[String] = {
    if (!isVisibleFromModule) {
      Some(s"operand '$this' is not visible from the current module ${Builder.currentModule.get.name}")
    } else {
      visibleFromBlock.map(MonoConnect.escapedScopeErrorMsg(this, _))
    }
  }

  // Internal API: returns a ref that can be assigned to, if consistent with the binding
  private[chisel3] def lref(implicit info: SourceInfo): Node = {
    requireIsHardware(this)
    requireVisible()
    topBindingOpt match {
      case Some(binding: ReadOnlyBinding) =>
        throwException(s"internal error: attempted to generate LHS ref to ReadOnlyBinding $binding")
      case Some(ViewBinding(target1, wr1)) =>
        val (target2, wr2) = reify(target1)
        val writability = wr1.combine(wr2)
        writability.reportIfReadOnly(target2.lref)(Wire(chiselTypeOf(target2)).lref)
      case Some(binding: TopBinding) => Node(this)
      case opt                       => throwException(s"internal error: unknown binding $opt in generating LHS ref")
    }
  }

  // Internal API: returns a ref, if bound
  private[chisel3] def ref(implicit info: SourceInfo): Arg = {
    def materializeWire(makeConst: Boolean = false): Arg = {
      if (!Builder.currentModule.isDefined) throwException(s"internal error: cannot materialize ref for $this")
      implicit val sourceInfo = UnlocatableSourceInfo
      if (makeConst) {
        WireDefault(Const(chiselTypeOf(this)), this).ref
      } else {
        WireDefault(this).ref
      }
    }
    requireIsHardware(this)
    topBindingOpt match {
      // DataView
      case Some(ViewBinding(target, _)) => reify(target)._1.ref
      case Some(_: AggregateViewBinding) =>
        reifyIdentityView(this) match {
          // If this is an identity view (a view of something of the same type), return ref of target
          case Some((target, _)) => target.ref
          // Otherwise, we need to materialize hardware of the correct type
          case _ => materializeWire()
        }
      // Literals
      case Some(ElementLitBinding(litArg)) => litArg
      case Some(BundleLitBinding(litMap)) =>
        litMap.get(this) match {
          case Some(litArg) => litArg
          case _            => materializeWire(true) // FIXME FIRRTL doesn't have Bundle literal expressions
        }
      case Some(VecLitBinding(litMap)) =>
        litMap.get(this) match {
          case Some(litArg) => litArg
          case _            => materializeWire(true) // FIXME FIRRTL doesn't have Vec literal expressions
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
  private[chisel3] def setAllParents(parent: Option[BaseModule]): Unit =
    DataMirror.collectAllMembers(this).foreach { x => x._parent = parent }

  private[chisel3] def width:                                                      Width
  private[chisel3] def firrtlConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit

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
    * Directionality data and probe information is still preserved.
    */
  private[chisel3] def cloneTypeFull: this.type = {
    val clone: this.type = this.cloneType // get a fresh object, without bindings
    // Only the top-level direction needs to be fixed up, cloneType should do the rest
    clone.specifiedDirection = specifiedDirection
    probe.setProbeModifier(clone, probeInfo)
    clone.isConst = isConst
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
  final def :=(that: => Data)(implicit sourceInfo: SourceInfo): Unit = {
    prefix(this) {
      this.connect(that)(sourceInfo)
    }
  }

  /** The "bulk connect operator", assigning elements in this Vec from elements in a Vec.
    *
    * For chisel3._, uses the `chisel3.internal.BiConnect` algorithm; sub-elements of that` may end up driving sub-elements of `this`
    *  - Complicated semantics, hard to write quickly, will likely be deprecated in the future
    *
    * For Chisel._, emits the FIRRTL.<- operator
    *  - Equivalent to `this :<>= that` without the restrictions that bundle field names and vector sizes must match
    *
    * @param that the Data to connect from
    * @group connection
    */
  final def <>(that: => Data)(implicit sourceInfo: SourceInfo): Unit = {
    prefix(this) {
      this.bulkConnect(that)(sourceInfo)
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

  private[chisel3] def _asTypeOfImpl[T <: Data](that: T)(implicit sourceInfo: SourceInfo): T = {
    that._fromUInt(this.asUInt).asInstanceOf[T].viewAsReadOnly { _ =>
      "Return values of asTypeOf are now read-only"
    }
  }

  /** Return a value of this type from a UInt type. Internal implementation for asTypeOf.
    *
    * Protected so that it can be implemented by the external FixedPoint library
    */
  protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data

  // Package private alias for _fromUInt so we can call it elsewhere in chisel3
  private[chisel3] final def _fromUIntPrivate(that: UInt)(implicit sourceInfo: SourceInfo): Data = _fromUInt(that)

  // The actual implementation of do_asUInt
  // @param first exists because of awkward behavior in Aggregate that requires changing 0.U to be zero-width to fix
  private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt

  protected def _asUIntImpl(implicit sourceInfo: SourceInfo): UInt = this._asUIntImpl(true)

  /** Default pretty printing */
  def toPrintable: Printable

  /** A non-ambiguous name of this `Data` for use in generated Verilog names */
  def typeName: String = simpleClassName(this.getClass)
}

object Data {
  // Needed for the `implicit def toConnectableDefault`
  import scala.language.implicitConversions

  private[chisel3] case class ProbeInfo(val writable: Boolean, color: Option[layer.Layer])

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
  implicit def toConnectableDefault[T <: Data](d: T): Connectable[T] = makeConnectableDefault(d)

  /** Create the default [[Connectable]] used for all instances of a [[Data]] of type T.
    *
    * This uses the default [[connectable.Connectable.apply]] as a starting point.
    *
    * Users can extend the [[HasCustomConnectable]] trait on any [[Data]] to further customize the [[Connectable]]. This
    * is checked for in any potentially nested [[Data]] and any customizations are applied on top of the default
    * [[Connectable]].
    */
  private[chisel3] def makeConnectableDefault[T <: Data](d: T): Connectable[T] = {
    val base = Connectable.apply(d)
    DataMirror
      .collectMembers(d) { case hasCustom: HasCustomConnectable =>
        hasCustom
      }
      .foldLeft(base)((connectable, hasCustom) => hasCustom.customConnectable(connectable))
  }

  /** Typeclass implementation of HasMatchingZipOfChildren for Data
    *
    * The canonical API to iterate through two Chisel types or components, where
    *   matching children are provided together, while non-matching members are provided
    *   separately
    *
    * Only zips immediate children (vs members, which are all children/grandchildren etc.)
    */
  implicit val dataMatchingZipOfChildren: DataMirror.HasMatchingZipOfChildren[Data] =
    new DataMirror.HasMatchingZipOfChildren[Data] {

      implicit class VecOptOps(vOpt: Option[Vec[Data]]) {
        // Like .get, but its already defined on Option
        def grab(i: Int): Option[Data] = vOpt.flatMap { _.lift(i) }
        def size = vOpt.map(_.size).getOrElse(0)
      }
      implicit class RecordOptGet(rOpt: Option[Record]) {
        // Like .get, but its already defined on Option
        def grab(k: String): Option[Data] = rOpt.flatMap { _._elements.get(k) }
        def keys:            Iterable[String] = rOpt.map { r => r._elements.map(_._1) }.getOrElse(Seq.empty[String])
      }
      // TODO(azidar): Rewrite this to be more clear, probably not the cleanest way to express this
      private def isDifferent(l: Option[Data], r: Option[Data]): Boolean =
        l.nonEmpty && r.nonEmpty && !isRecord(l, r) && !isVec(l, r) && !isElement(l, r) && !isProbe(l, r)
      private def isRecord(l: Option[Data], r: Option[Data]): Boolean =
        l.orElse(r).map { _.isInstanceOf[Record] }.getOrElse(false)
      private def isVec(l: Option[Data], r: Option[Data]): Boolean =
        l.orElse(r).map { _.isInstanceOf[Vec[_]] }.getOrElse(false)
      private def isElement(l: Option[Data], r: Option[Data]): Boolean =
        l.orElse(r).map { _.isInstanceOf[Element] }.getOrElse(false)
      private def isProbe(l: Option[Data], r: Option[Data]): Boolean =
        l.orElse(r).map { x => x.isInstanceOf[Data] && DataMirror.hasProbeTypeModifier(x) }.getOrElse(false)

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
          case (lOpt, rOpt) if isProbe(lOpt, rOpt)     => Nil
          case (lOpt: Option[Vec[Data] @unchecked], rOpt: Option[Vec[Data] @unchecked]) if isVec(lOpt, rOpt) =>
            (0 until (lOpt.size.max(rOpt.size))).map { i => (lOpt.grab(i), rOpt.grab(i)) }
          case (lOpt: Option[Record @unchecked], rOpt: Option[Record @unchecked]) if isRecord(lOpt, rOpt) =>
            (lOpt.keys ++ rOpt.keys).toList.distinct.map { k => (lOpt.grab(k), rOpt.grab(k)) }
          case (lOpt: Option[Element @unchecked], rOpt: Option[Element @unchecked]) if isElement(lOpt, rOpt) => Nil
          case _ =>
            throw new InternalErrorException(s"Match Error: left=$left, right=$right")
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
  implicit class DataEquality[T <: Data](lhs: T)(implicit sourceInfo: SourceInfo) {

    /** Dynamic recursive equality operator for generic [[Data]]
      *
      * @param rhs a hardware [[Data]] to compare `lhs` to
      * @return a hardware [[Bool]] asserted if `lhs` is equal to `rhs`
      * @throws ChiselException when `lhs` and `rhs` are different types during elaboration time
      */
    def ===(rhs: T): Bool = {
      (lhs, rhs) match {
        case (thiz: UInt, that: UInt)             => thiz === that
        case (thiz: SInt, that: SInt)             => thiz === that
        case (thiz: AsyncReset, that: AsyncReset) => thiz.asBool === that.asBool
        case (thiz: Reset, that: Reset)           => thiz.asBool === that.asBool
        case (thiz: EnumType, that: EnumType)     => thiz === that
        case (thiz: Clock, that: Clock)           => thiz.asUInt === that.asUInt
        case (thiz: Vec[_], that: Vec[_]) =>
          if (thiz.length != that.length) {
            throwException(s"Cannot compare Vecs $thiz and $that: Vec sizes differ")
          } else {
            thiz.elementsIterator
              .zip(that.elementsIterator)
              .map { case (thisData, thatData) => thisData === thatData }
              .reduceOption(_ && _) // forall but that isn't defined for Bool on Seq
              .getOrElse(true.B)
          }
        case (thiz: Record, that: Record) =>
          if (thiz._elements.size != that._elements.size) {
            throwException(s"Cannot compare Bundles $thiz and $that: Bundle types differ")
          } else {
            thiz._elements.map { case (thisName, thisData) =>
              if (!that._elements.contains(thisName))
                throwException(
                  s"Cannot compare Bundles $thiz and $that: field $thisName (from $thiz) was not found in $that"
                )

              val thatData = that._elements(thisName)

              try {
                thisData === thatData
              } catch {
                case e: chisel3.ChiselException =>
                  throwException(
                    s"Cannot compare field $thisName in Bundles $thiz and $that: ${e.getMessage.split(": ").last}"
                  )
              }
            }
              .reduceOption(_ && _) // forall but that isn't defined for Bool on Seq
              .getOrElse(true.B)
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

  implicit class AsReadOnly[T <: Data](self: T) {

    /** Returns a read-only view of this Data
      *
      * It is illegal to connect to the return value of this method.
      * This Data this method is called on must be a hardware type.
      */
    def readOnly(implicit sourceInfo: SourceInfo): T = {
      val alreadyReadOnly = self.isLit || self.topBindingOpt.exists(_.isInstanceOf[ReadOnlyBinding])
      if (alreadyReadOnly) {
        self
      } else {
        self.viewAsReadOnly(_ => "Cannot connect to read-only value")
      }
    }
  }
}

trait WireFactory {

  /** Construct a [[Wire]] from a type template
    * @param t The template from which to construct this wire
    */
  def apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = {
    val prevId = Builder.idGen.value
    val t = source // evaluate once (passed by name)
    requireIsChiselType(t, "wire type")

    val x = if (!t.mustClone(prevId)) t else t.cloneTypeFull

    // Bind each element of x to being a Wire
    x.bind(WireBinding(Builder.forcedUserModule, Builder.currentBlock))

    pushCommand(DefWire(sourceInfo, x))

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

private[chisel3] sealed trait WireDefaultImpl {

  private def applyImpl[T <: Data](
    t:    T,
    init: Data
  )(
    implicit sourceInfo: SourceInfo
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
    implicit sourceInfo: SourceInfo
  ): T = {
    applyImpl(t, init)
  }

  /** Construct a [[Wire]] with a type template and a default connection
    * @param t The type template used to construct this [[Wire]]
    * @param init The hardware value that will serve as the default value
    */
  def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo): T = {
    applyImpl(t, init)
  }

  /** Construct a [[Wire]] with a default connection
    * @param init The hardware value that will serve as a type template and default value
    */
  def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo): T = {
    val model = (init match {
      // If init is a literal without forced width OR any non-literal, let width be inferred
      case init: Bits if !init.litIsForcedWidth.getOrElse(false) => init.cloneTypeWidth(Width())
      case _ => init.cloneTypeFull
    }).asInstanceOf[T]
    apply(model, init)
  }
}

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
  */
object WireDefault extends WireDefaultImpl

/** Utility for constructing hardware wires with a default connection
  *
  * Alias for [[WireDefault]].
  *
  * @note The `Init` in `WireInit` refers to a "default" connection. This is in contrast to
  * [[RegInit]] where the `Init` refers to a value on reset.
  */
object WireInit extends WireDefaultImpl

/** RHS (source) for Invalidate API.
  * Causes connection logic to emit a DefInvalid when connected to an output port (or wire).
  */
final case object DontCare extends Element with connectable.ConnectableDocs {
  // This object should be initialized before we execute any user code that refers to it,
  //  otherwise this "Chisel" object will end up on the UserModule's id list.
  // We make it private to chisel3 so it has to be accessed through the package object.

  private[chisel3] override val width: Width = UnknownWidth

  bind(DontCareBinding(), SpecifiedDirection.Output)
  override def cloneType: this.type = DontCare

  override def toString: String = "DontCare()"

  override def litOption: Option[BigInt] = None

  def toPrintable: Printable = PString("DONTCARE")

  override protected def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data = {
    Builder.error("DontCare cannot be a connection sink (LHS)")
    this
  }

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = {
    Builder.error("DontCare does not have a UInt representation")
    0.U
  }

  /** $colonGreaterEq
    *
    * @group connection
    * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
    */
  final def :>=[T <: Data](producer: => T)(implicit sourceInfo: SourceInfo): Unit =
    this.asInstanceOf[Data] :>= producer.asInstanceOf[Data]
}

/** Trait to indicate that a subclass of [[Data]] has a custom [[Connectable]].
  *
  * Users can implement the [[customConnectable]] method, which receives a default [[Connectable]], and is expected to
  * use the methods on [[Connectable]] to customize it. For example, a [[Bundle]] could define this by using
  * [[connectable.Connectable.exclude(members*]] to always exlude a specific member:
  *
  *  {{{
  *    class MyBundle extends Bundle with HasCustomConnectable {
  *      val foo = Bool()
  *      val bar = Bool()
  *
  *      override def customConnectable[T <: Data](base: Connectable[T]): Connectable[T] = {
  *        base.exclude(_ => bar)
  *      }
  *    }
  *  }}}
  */
trait HasCustomConnectable { this: Data =>
  def customConnectable[T <: Data](base: Connectable[T]): Connectable[T]
}
