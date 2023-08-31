// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.internal.firrtl.{LitArg, PropertyLit}
import chisel3.properties.Class

import scala.collection.immutable.VectorMap

@deprecated(deprecatedPublicAPIMsg + ". Use chisel3.experimental.requireIsHardware instead", "Chisel 3.6")
object requireIsHardware {
  def apply(node: Data, msg: String = ""): Unit = chisel3.experimental.requireIsHardware.apply(node, msg)
}

@deprecated(deprecatedPublicAPIMsg + ". Use chisel3.experimental.requireIsChiselType instead", "Chisel 3.6")
object requireIsChiselType {
  def apply(node: Data, msg: String = ""): Unit = chisel3.experimental.requireIsChiselType.apply(node, msg)
}

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
sealed trait Binding {
  def location: Option[BaseModule]
}
// Top-level binding representing hardware, not a pointer to another binding (like ChildBinding)
sealed trait TopBinding extends Binding

// Constrained-ness refers to whether 'bound by Module boundaries'
// An unconstrained binding, like a literal, can be read by everyone
sealed trait UnconstrainedBinding extends TopBinding {
  def location: Option[BaseModule] = None
}
// A constrained binding can only be read/written by specific modules
// Location will track where this Module is, and the bound object can be referenced in FIRRTL
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
sealed trait ReadOnlyBinding extends TopBinding

// A component that can potentially be declared inside a 'when'
sealed trait ConditionalDeclarable extends TopBinding {
  def visibility: Option[WhenContext]
}

// TODO(twigg): Ops between unenclosed nodes can also be unenclosed
// However, Chisel currently binds all op results to a module
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding

// Added to handle BoringUtils in Chisel
private[chisel3] case class SecretPortBinding(enclosure: BaseModule) extends ConstrainedBinding

@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class OpBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ReadOnlyBinding
    with ConditionalDeclarable
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class MemoryPortBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ConditionalDeclarable
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class RegBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ConditionalDeclarable
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class WireBinding(enclosure: RawModule, visibility: Option[WhenContext])
    extends ConstrainedBinding
    with ConditionalDeclarable

private[chisel3] case class ClassBinding(enclosure: Class) extends ConstrainedBinding with ReadOnlyBinding

private[chisel3] case class ObjectFieldBinding(enclosure: BaseModule) extends ConstrainedBinding

@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class ChildBinding(parent: Data) extends Binding {
  def location: Option[BaseModule] = parent.topBinding.location
}

/** Special binding for Vec.sample_element */
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class SampleElementBinding[T <: Data](parent: Vec[T]) extends Binding {
  def location = parent.topBinding.location
}

/** Special binding for Mem types */
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class MemTypeBinding[T <: Data](parent: MemBase[T]) extends Binding {
  def location: Option[BaseModule] = parent._parent
}
// A DontCare element has a specific Binding, somewhat like a literal.
// It is a source (RHS). It may only be connected/applied to sinks.
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class DontCareBinding() extends UnconstrainedBinding

// Views currently only support 1:1 Element-level mappings
private[chisel3] case class ViewBinding(target: Element) extends UnconstrainedBinding

/** Binding for Aggregate Views
  * @param childMap Mapping from children of this view to their respective targets
  * @param target Optional Data this Aggregate views if the view is total and the target is a Data
  * @note For any Elements in the childMap, both key and value must be Elements
  * @note The types of key and value need not match for the top Data in a total view of type
  *       Aggregate
  */
private[chisel3] case class AggregateViewBinding(childMap: Map[Data, Data]) extends UnconstrainedBinding {
  // Helper lookup function since types of Elements always match
  def lookup(key: Element): Option[Element] = childMap.get(key).map(_.asInstanceOf[Element])
}

/** Binding for Data's returned from accessing an Instance/Definition members, if not readable/writable port */
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
private[chisel3] case object CrossModuleBinding extends TopBinding {
  def location = None
}

sealed trait LitBinding extends UnconstrainedBinding with ReadOnlyBinding
// Literal binding attached to a element that is not part of a Bundle.
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class ElementLitBinding(litArg: LitArg) extends LitBinding
// Literal binding attached to the root of a Bundle, containing literal values of its children.
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class BundleLitBinding(litMap: Map[Data, LitArg]) extends LitBinding
// Literal binding attached to the root of a Vec, containing literal values of its children.
@deprecated(deprecatedPublicAPIMsg, "Chisel 3.6")
case class VecLitBinding(litMap: VectorMap[Data, LitArg]) extends LitBinding
// Literal binding attached to a Property.
private[chisel3] case object PropertyValueBinding extends UnconstrainedBinding with ReadOnlyBinding
