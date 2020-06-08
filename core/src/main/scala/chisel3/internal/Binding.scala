// See LICENSE for license details.

package chisel3.internal

import chisel3._
import chisel3.experimental.BaseModule
import chisel3.internal.firrtl.LitArg

/** Requires that a node is hardware ("bound")
  */
object requireIsHardware {
  def apply(node: Data, msg: String = ""): Unit = {
    node._parent match {  // Compatibility layer hack
      case Some(x: BaseModule) => x._compatAutoWrapPorts
      case _ =>
    }
    if (!node.isSynthesizable) {
      val prefix = if (msg.nonEmpty) s"$msg " else ""
      throw ExpectedHardwareException(s"$prefix'$node' must be hardware, " +
        "not a bare Chisel type. Perhaps you forgot to wrap it in Wire(_) or IO(_)?")
    }
  }
}

/** Requires that a node is a chisel type (not hardware, "unbound")
  */
object requireIsChiselType {
  def apply(node: Data, msg: String = ""): Unit = if (node.isSynthesizable) {
    val prefix = if (msg.nonEmpty) s"$msg " else ""
    throw ExpectedChiselTypeException(s"$prefix'$node' must be a Chisel type, not hardware")
  }
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
      case PortBinding(_) => direction match {
        case ActualDirection.Output => Output
        case ActualDirection.Input => Input
        case dir => throw new RuntimeException(s"Unexpected port element direction '$dir'")
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
      case None => Some(enclosure)
      case Some(aspect) => Some(aspect)
    }
  }
}

// A binding representing a data that cannot be (re)assigned to.
sealed trait ReadOnlyBinding extends TopBinding

// TODO(twigg): Ops between unenclosed nodes can also be unenclosed
// However, Chisel currently binds all op results to a module
case class OpBinding(enclosure: RawModule) extends ConstrainedBinding with ReadOnlyBinding
case class MemoryPortBinding(enclosure: RawModule) extends ConstrainedBinding
case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding
case class RegBinding(enclosure: RawModule) extends ConstrainedBinding
case class WireBinding(enclosure: RawModule) extends ConstrainedBinding

case class ChildBinding(parent: Data) extends Binding {
  def location: Option[BaseModule] = parent.topBinding.location
}
/** Special binding for Vec.sample_element */
case class SampleElementBinding[T <: Data](parent: Vec[T]) extends Binding {
  def location = parent.topBinding.location
}
// A DontCare element has a specific Binding, somewhat like a literal.
// It is a source (RHS). It may only be connected/applied to sinks.
case class DontCareBinding() extends UnconstrainedBinding

sealed trait LitBinding extends UnconstrainedBinding with ReadOnlyBinding
// Literal binding attached to a element that is not part of a Bundle.
case class ElementLitBinding(litArg: LitArg) extends LitBinding
// Literal binding attached to the root of a Bundle, containing literal values of its children.
case class BundleLitBinding(litMap: Map[Data, LitArg]) extends LitBinding
