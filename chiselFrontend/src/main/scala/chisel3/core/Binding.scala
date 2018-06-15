package chisel3.core

import chisel3.internal.Builder.{forcedModule}

object Binding {
  class BindingException(message: String) extends Exception(message)
  /** A function expected a Chisel type but got a hardware object
    */
  case class ExpectedChiselTypeException(message: String) extends BindingException(message)
  /**A function expected a hardware object but got a Chisel type
    */
  case class ExpectedHardwareException(message: String) extends BindingException(message)
  /** An aggregate had a mix of specified and unspecified directionality children
    */
  case class MixedDirectionAggregateException(message: String) extends BindingException(message)
  /** Attempted to re-bind an already bound (directionality or hardware) object
    */
  case class RebindingException(message: String) extends BindingException(message)
}

/** Requires that a node is hardware ("bound")
  */
object requireIsHardware {
  def apply(node: Data, msg: String = "") = {
    node._parent match {  // Compatibility layer hack
      case Some(x: BaseModule) => x._compatAutoWrapPorts
      case _ =>
    }
    if (!node.hasBinding) {
      val prefix = if (msg.nonEmpty) s"$msg " else ""
      throw Binding.ExpectedHardwareException(s"$prefix'$node' must be hardware, " +
        "not a bare Chisel type. Perhaps you forgot to wrap it in Wire(_) or IO(_)?")
    }
  }
}

/** Requires that a node is a chisel type (not hardware, "unbound")
  */
object requireIsChiselType {
  def apply(node: Data, msg: String = "") = if (node.hasBinding) {
    val prefix = if (msg.nonEmpty) s"$msg " else ""
    throw Binding.ExpectedChiselTypeException(s"$prefix'$node' must be a Chisel type, not hardware")
  }
}

// Element only direction used for the Binding system only.
sealed abstract class BindingDirection
object BindingDirection {
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
  def from(binding: TopBinding, direction: ActualDirection) = {
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
  def location = None
}
// A constrained binding can only be read/written by specific modules
// Location will track where this Module is
sealed trait ConstrainedBinding extends TopBinding {
  def enclosure: BaseModule
  def location = Some(enclosure)
}

// A binding representing a data that cannot be (re)assigned to.
sealed trait ReadOnlyBinding extends TopBinding

// TODO literal info here
case class LitBinding() extends UnconstrainedBinding with ReadOnlyBinding
// TODO(twigg): Ops between unenclosed nodes can also be unenclosed
// However, Chisel currently binds all op results to a module
case class OpBinding(enclosure: UserModule) extends ConstrainedBinding with ReadOnlyBinding
case class MemoryPortBinding(enclosure: UserModule) extends ConstrainedBinding
case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding
case class RegBinding(enclosure: UserModule) extends ConstrainedBinding
case class WireBinding(enclosure: UserModule) extends ConstrainedBinding

case class ChildBinding(parent: Data) extends Binding {
  def location = parent.binding.location
}
// A DontCare element has a specific Binding, somewhat like a literal.
// It is a source (RHS). It may only be connected/applied to sinks.
case class DontCareBinding() extends UnconstrainedBinding
