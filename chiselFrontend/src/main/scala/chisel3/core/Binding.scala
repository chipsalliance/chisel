package chisel3.core

import chisel3.internal.Builder.{forcedModule}

/**
 * The purpose of a Binding is to indicate what type of hardware 'entity' a
 * specific Data's leaf Elements is actually bound to. All Data starts as being
 * Unbound (and the whole point of cloneType is to return an unbound version).
 * Then, specific API calls take a Data, and return a bound version (either by
 * binding the original model or cloneType then binding the clone). For example,
 * Reg[T<:Data](...) returns a T bound to RegBinding.
 *
 * It is considered invariant that all Elements of a single Data are bound to
 * the same concrete type of Binding.
 *
 * These bindings can be checked (e.g. checkSynthesizable) to make sure certain
 * operations are valid. For example, arithemetic operations or connections can
 * only be executed between synthesizable nodes. These checks are to avoid
 * undefined reference errors.
 *
 * Bindings can carry information about the particular element in the graph it
 * represents like:
 * - For ports (and unbound), the 'direction'
 * - For (relevant) synthesizable nodes, the enclosing Module
 *
 * TODO(twigg): Enrich the bindings to carry more information like the hosting
 * module (when applicable), direction (when applicable), literal info (when
 * applicable). Can ensure applicable data only stored on relevant nodes. e.g.
 * literal info on LitBinding, direction info on UnboundBinding and PortBinding,
 * etc.
 *
 * TODO(twigg): Currently, bindings only apply at the Element level and an
 * Aggregate is considered bound via its elements. May be appropriate to allow
 * Aggregates to be bound along with the Elements. However, certain literal and
 * port direction information doesn't quite make sense in aggregates. This would
 * elegantly handle the empty Vec or Record problem though.
 *
 * TODO(twigg): Binding is currently done via allElements. It may be more
 * elegant if this was instead done as a more explicit tree walk as that allows
 * for better errors.
 */

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

// Constrained-ness refers to whether 'bound by Module boundaries'
// An unconstrained binding, like a literal, can be read by everyone
sealed trait UnconstrainedBinding extends Binding {
  def location = None
}
// A constrained binding can only be read/written by specific modules
// Location will track where this Module is
sealed trait ConstrainedBinding extends Binding {
  def enclosure: BaseModule
  def location = Some(enclosure)
}

case class LitBinding() // will eventually have literal info
    extends Binding with UnconstrainedBinding

case class MemoryPortBinding(enclosure: UserModule)
    extends Binding with ConstrainedBinding

// TODO(twigg): Ops between unenclosed nodes can also be unenclosed
// However, Chisel currently binds all op results to a module
case class OpBinding(enclosure: UserModule)
    extends Binding with ConstrainedBinding

case class PortBinding(enclosure: BaseModule)
    extends Binding with ConstrainedBinding

case class RegBinding(enclosure: UserModule)
    extends Binding with ConstrainedBinding

case class WireBinding(enclosure: UserModule)
    extends Binding with ConstrainedBinding

case class ChildBinding(parent: Data)
    extends Binding {
  def location = parent.binding.location
}
