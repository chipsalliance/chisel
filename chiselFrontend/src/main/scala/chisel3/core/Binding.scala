package chisel3.core

import chisel3.internal.Builder.{forcedModule}

object Binding {
  case class BindingException(message: String) extends Exception(message)
  def MissingIOWrapperException = BindingException(": Missing IO() wrapper")
}

/** Requires that a node is hardware ("bound")
  */
object requireIsHardware {
  def apply(node: Data) = if (!node.hasBinding) {
    throw new Binding.BindingException(s"'$node' must be hardware, not a bare Chisel type")
  }
  def apply(node: Data, msg: String) = if (!node.hasBinding) {
    throw new Binding.BindingException(s"$msg '$node' must be hardware, not a bare Chisel type")
  }
}

/** Requires that a node is a chisel type (not hardware, "unbound")
  */
object requireIsChiselType {
  def apply(node: Data) = if (node.hasBinding) {
    throw new Binding.BindingException(s"'$node' must be a Chisel type, not hardware")
  }
  def apply(node: Data, msg: String) = if (node.hasBinding) {
    throw new Binding.BindingException(s"$msg '$node' must be a Chisel type, not hardware")
  }
}

// Element only direction used for the Binding system only.
sealed abstract class BindingDirection
object BindingDirection {
  case object Internal extends BindingDirection  // internal type or wire
  case object Output extends BindingDirection  // module port
  case object Input extends BindingDirection  // module port

  def from(binding: TopBinding, direction: ActualDirection) = {
    binding match {
      case PortBinding(_) => direction match {
        case ActualDirection.Output => Output
        case ActualDirection.Input => Input
        case _ => throw new RuntimeException("Unexpected port element direction")
      }
      case _ => Internal
    }
  }
}

// Location refers to 'where' in the Module hierarchy this lives
sealed trait Binding {
  def location: Option[BaseModule]
}
// Top-level binding representing hardware, not a pointer to another binding
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

// TODO literal info here
case class LitBinding() extends UnconstrainedBinding
// TODO(twigg): Ops between unenclosed nodes can also be unenclosed
// However, Chisel currently binds all op results to a module
case class OpBinding(enclosure: UserModule) extends ConstrainedBinding
case class MemoryPortBinding(enclosure: UserModule) extends ConstrainedBinding
case class PortBinding(enclosure: BaseModule) extends ConstrainedBinding
case class RegBinding(enclosure: UserModule) extends ConstrainedBinding
case class WireBinding(enclosure: UserModule) extends ConstrainedBinding

case class ChildBinding(parent: Data) extends Binding {
  def location = parent.binding.location
}
