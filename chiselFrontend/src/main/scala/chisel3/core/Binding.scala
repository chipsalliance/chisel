package chisel3.core

import chisel3.internal.Builder.{forcedModule}

/** Requires that a node is hardware ("bound")
  */
object requireIsHardware {
  def apply(node: Data) = require(node.hasBinding)
  def apply(node: Data, msg: String) = require(node.hasBinding, msg)
}

/** Requires that a node is a chisel type (not hardware, "unbound")
  */
object requireIsChiselType {
  def apply(node: Data) = require(!node.hasBinding)
  def apply(node: Data, msg: String) = require(!node.hasBinding, msg)
}


object Binding {
  case class BindingException(message: String) extends Exception(message)
  def AlreadyBoundException(binding: String) = BindingException(s": Already bound to $binding")
  def NotSynthesizableException = BindingException(s": Not bound to synthesizable node, currently only Type description")
  def MissingIOWrapperException = BindingException(": Missing IO() wrapper")
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
