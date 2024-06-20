// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3._
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.firrtl.ir.{LitArg, PropertyLit}
import chisel3.properties.Class

import scala.collection.immutable.VectorMap

// Element only direction used for the Binding system only.
private[chisel3] sealed abstract class BindingDirection
private[chisel3] object BindingDirection {

  /** Internal type or wire
    */
  case object Internal extends BindingDirection

  /** Module port with output direction
    */
  case object Output extends BindingDirection

  /** Module port with input direction
    */
  case object Input extends BindingDirection

  /** Determine the BindingDirection of an Element given its top binding and resolved direction.
    */
  def from(binding: TopBinding, direction: ActualDirection): BindingDirection = {
    binding match {
      case _: PortBinding | _: SecretPortBinding =>
        direction match {
          case ActualDirection.Output => Output
          case ActualDirection.Input  => Input
          case dir                    => throw new RuntimeException(s"Unexpected port element direction '$dir'")
        }
      case _ => Internal
    }
  }
}

// Location refers to 'where' in the Module hierarchy this lives
@deprecated(deprecatedPublicAPIMsg, "Chisel 6.0")
sealed trait Binding {
  def location: Option[BaseModule]
}
// Top-level binding representing hardware, not a pointer to another binding (like ChildBinding)
@deprecated(deprecatedPublicAPIMsg, "Chisel 6.0")
sealed trait TopBinding extends Binding

// Constrained-ness refers to whether 'bound by Module boundaries'
// An unconstrained binding, like a literal, can be read by everyone
@deprecated(deprecatedPublicAPIMsg, "Chisel 6.0")
sealed trait UnconstrainedBinding extends TopBinding {
  def location: Option[BaseModule] = None
}
// A constrained binding can only be read/written by specific modules
// Location will track where this Module is, and the bound object can be referenced in FIRRTL
@deprecated(deprecatedPublicAPIMsg, "Chisel 6.0")
sealed trait ConstrainedBinding extends TopBinding {
  def enclosure: BaseModule
  def location: Option[BaseModule] = {
    // If an aspect is present, return the aspect module. Otherwise, return the enclosure module
    // This allows aspect modules to pretend to be enclosed modules for connectivity checking,
    // inside vs outside instance checking, etc.
    Builder.aspectModule(enclosure) match {
      case None         => Some(enclosure)
      case Some(aspect) => Some(aspect)
    }
  }
}

// A binding representing a data that cannot be (re)assigned to.
@deprecated(deprecatedPublicAPIMsg, "Chisel 6.0")
sealed trait ReadOnlyBinding extends TopBinding

// A component that can potentially be declared inside a 'when'
@deprecated(deprecatedPublicAPIMsg, "Chisel 6.0")
sealed trait ConditionalDeclarable extends TopBinding {
  def visibility: Option[WhenContext]
}

// TODO(twigg): Ops between unenclosed nodes can also be unenclosed
// However, Chisel currently binds all op results to a module
private[chisel3] case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding

// Added to handle BoringUtils in Chisel
private[chisel3] case class SecretPortBinding(enclosure: BaseModule) extends ConstrainedBinding

private[chisel3] case class OpBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ReadOnlyBinding
    with ConditionalDeclarable
private[chisel3] case class MemoryPortBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ConditionalDeclarable
private[chisel3] case class RegBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ConditionalDeclarable
private[chisel3] case class WireBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ConditionalDeclarable

private[chisel3] case class ClassBinding(enclosure: Class) extends ConstrainedBinding with ReadOnlyBinding

private[chisel3] case class ObjectFieldBinding(enclosure: BaseModule) extends ConstrainedBinding

private[chisel3] case class ChildBinding(parent: Data) extends Binding {
  def location: Option[BaseModule] = parent.topBinding.location
}

/** Special binding for Vec.sample_element */
private[chisel3] case class SampleElementBinding[T <: Data](parent: Vec[T]) extends Binding {
  def location = parent.topBinding.location
}

/** Special binding for Mem types */
private[chisel3] case class MemTypeBinding[T <: Data](parent: MemBase[T]) extends Binding {
  def location: Option[BaseModule] = parent._parent
}
// A DontCare element has a specific Binding, somewhat like a literal.
// It is a source (RHS). It may only be connected/applied to sinks.
private[chisel3] case class DontCareBinding() extends UnconstrainedBinding

/** Views are able to restrict writability of the target */
private[chisel3] sealed trait ViewWriteability {

  final def isReadOnly: Boolean = this != ViewWriteability.Default

  /** Report errors if this is a read-only view
    *
    * Will use onPass value if normal operation can continue.
    * Will use onFail value if this is a read-only view to continue elaboration and aggregate more errors.
    */
  final def reportIfReadOnly[A](onPass: => A)(onFail: => A)(implicit info: SourceInfo): A = this match {
    case ViewWriteability.Default => onPass
    case ViewWriteability.ReadOnlyDeprecated(getWarning) =>
      Builder.warning(getWarning(info))
      onPass // This is just a warning so we propagate the pass value.
    case ViewWriteability.ReadOnly(getError) =>
      Builder.error(getError(info))
      onFail
  }

  final def reportIfReadOnlyUnit(onPass: => Unit)(implicit info: SourceInfo): Unit =
    reportIfReadOnly[Unit](onPass)(())

  /** Combine two writabilities into one */
  def combine(that: ViewWriteability): ViewWriteability
}
private[chisel3] object ViewWriteability {

  /** Default is no modification, writability of target applies */
  case object Default extends ViewWriteability {
    override def combine(that: ViewWriteability): ViewWriteability = that
  }

  /** Signals that will eventually become read only */
  case class ReadOnlyDeprecated(getWarning: SourceInfo => Warning) extends ViewWriteability {
    override def combine(that: ViewWriteability): ViewWriteability = that match {
      case ro: ReadOnly => ro
      case _ => this
    }
  }

  /** Signals that are read only */
  case class ReadOnly(getError: SourceInfo => String) extends ViewWriteability {
    override def combine(that: ViewWriteability): ViewWriteability = this
  }
}

// Views currently only support 1:1 Element-level mappings
private[chisel3] case class ViewBinding(target: Element, writability: ViewWriteability)
    extends Binding
    with ConditionalDeclarable {
  def location: Option[BaseModule] = target.binding.flatMap(_.location)
  def visibility: Option[WhenContext] = target.binding.flatMap {
    case c: ConditionalDeclarable => c.visibility
    case _ => None
  }
}

/** Binding for Aggregate Views
  * @param childMap Mapping from children of this view to their respective targets
  * @param writabilityMap Information about writability of this Aggregate or its children.
  *   None means all children are writable. This Map is hierarchical, access should go through `lookupWritability`.
  * @note For any Elements in the childMap, both key and value must be Elements
  * @note The types of key and value need not match for the top Data in a total view of type
  *       Aggregate
  */
private[chisel3] case class AggregateViewBinding(
  childMap:       Map[Data, Data],
  writabilityMap: Option[Map[Data, ViewWriteability]])
    extends Binding
    with ConditionalDeclarable {
  // Helper lookup function since types of Elements always match
  def lookup(key: Element): Option[Element] = childMap.get(key).map(_.asInstanceOf[Element])

  /** Lookup the writability of a member of this view.
    *
    * Use this instead of accessing the Map directly.
    */
  def lookupWritability(key: Data): ViewWriteability = {
    def rec(map: Map[Data, ViewWriteability], key: Data): ViewWriteability = {
      map.getOrElse(
        key, {
          key.binding match {
            case Some(ChildBinding(parent)) => rec(map, parent)
            case _                          => throwException(s"Internal error! $key not found in AggregateViewBinding writabilityMap!")
          }
        }
      )
    }
    writabilityMap.map(rec(_, key)).getOrElse(ViewWriteability.Default)
  }

  // FIXME Technically an AggregateViewBinding can have multiple locations and visibilities
  // Fixing this requires an overhaul to this code so for now we just do the best we can
  // Return a location if there is a unique one for all targets, None otherwise
  lazy val location: Option[BaseModule] = {
    val locations = childMap.values.view.flatMap(_.binding.toSeq.flatMap(_.location)).toVector.distinct
    if (locations.size == 1) Some(locations.head)
    else None
  }
  lazy val visibility: Option[WhenContext] = {
    val contexts = childMap.values.view
      .flatMap(_.binding.toSeq.collect { case c: ConditionalDeclarable => c.visibility }.flatten)
      .toVector
      .distinct
    if (contexts.size == 1) Some(contexts.head)
    else None
  }
}

/** Binding for Data's returned from accessing an Instance/Definition members, if not readable/writable port */
private[chisel3] case object CrossModuleBinding extends TopBinding {
  def location = None
}

@deprecated(deprecatedPublicAPIMsg, "Chisel 6.0")
sealed trait LitBinding extends UnconstrainedBinding with ReadOnlyBinding
// Literal binding attached to a element that is not part of a Bundle.
private[chisel3] case class ElementLitBinding(litArg: LitArg) extends LitBinding
// Literal binding attached to the root of a Bundle, containing literal values of its children.
private[chisel3] case class BundleLitBinding(litMap: Map[Data, LitArg]) extends LitBinding
// Literal binding attached to the root of a Vec, containing literal values of its children.
private[chisel3] case class VecLitBinding(litMap: VectorMap[Data, LitArg]) extends LitBinding
// Literal binding attached to a Property.
private[chisel3] case object PropertyValueBinding extends UnconstrainedBinding with ReadOnlyBinding
