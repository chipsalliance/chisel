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
  // Two bindings are 'compatible' if they are the same type.
  // Check currently kind of weird: just ensures same class
  private def compatible(a: Binding, b: Binding): Boolean = a.getClass == b.getClass
  private def compatible(nodes: Seq[Binding]): Boolean =
    if(nodes.size > 1)
      (for((a,b) <- nodes zip nodes.tail) yield compatible(a,b))
      .fold(true)(_&&_)
    else true

  case class BindingException(message: String) extends Exception(message)
  def AlreadyBoundException(binding: String) = BindingException(s": Already bound to $binding")
  def NotSynthesizableException = BindingException(s": Not bound to synthesizable node, currently only Type description")
  def MissingIOWrapperException = BindingException(": Missing IO() wrapper")

  // This recursively walks down the Data tree to look at all the leaf 'Element's
  // Will build up an error string in case something goes wrong
  // TODO(twigg): Make member function of Data.
  //   Allows oddities like sample_element to be better hidden
  private def walkToBinding(target: Data, checker: Element=>Unit): Unit = target match {
    case (element: Element) => checker(element)
    case (vec: Vec[Data @unchecked]) => {
      try walkToBinding(vec.sample_element, checker)
      catch {
        case BindingException(message) => throw BindingException(s"(*)$message")
      }
      for(idx <- 0 until vec.length) {
        try walkToBinding(vec(idx), checker)
        catch {
          case BindingException(message) => throw BindingException(s"($idx)$message")
        }
      }
    }
    case (record: Record) => {
      for((field, subelem) <- record.elements) {
        try walkToBinding(subelem, checker)
        catch {
          case BindingException(message) => throw BindingException(s".$field$message")
        }
      }
    }
  }

  // Use walkToBinding to actually rebind the node type
  def bind[T<:Data](target: T, binder: Binder[_<:Binding], error_prelude: String): target.type = {
    try walkToBinding(
      target,
      element => element.binding match {
        case unbound @ UnboundBinding(_) => {
          element.binding = binder(unbound)
        }
        case binding => throw AlreadyBoundException(binding.toString)
      }
    )
    catch {
      case BindingException(message) => throw BindingException(s"$error_prelude$message")
    }
    target
  }

  // Excepts if any root element is already bound
  def checkUnbound(target: Data, error_prelude: String): Unit = {
    try walkToBinding(
      target,
      element => element.binding match {
        case unbound @ UnboundBinding(_) => {}
        case binding => throw AlreadyBoundException(binding.toString)
      }
    )
    catch {
      case BindingException(message) => throw BindingException(s"$error_prelude$message")
    }
  }

  // Excepts if any root element is unbound and thus not on the hardware graph
  def checkSynthesizable(target: Data, error_prelude: String): Unit = {
    try walkToBinding(
      target,
      element => {
        // Compatibility mode to automatically wrap ports in IO
        // TODO: remove me, perhaps by removing Bindings checks from compatibility mode
        element._parent match {
          case Some(x: BaseModule) => x._autoWrapPorts
          case _ =>
        }
        // Actual binding check
        element.binding match {
          case SynthesizableBinding() => // OK
          case binding => {
            // Attempt to diagnose common bindings issues, like forgot to wrap IO(...)
            element._parent match {
              case Some(x: LegacyModule) =>
                // null check in case we try to access io before it is defined
                if ((x.io != null) && (x.io.flatten contains element)) {
                  throw MissingIOWrapperException
                }
              case _ =>
            }
            // Fallback generic exception
            throw NotSynthesizableException
          }
        }
      }
    )
    catch {
      case BindingException(message) => throw BindingException(s"$error_prelude$message")
    }
  }
}

// Location refers to 'where' in the Module hierarchy this lives
sealed trait Binding {
  def location: Option[BaseModule]
  def direction: Option[Direction]
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

// An undirectioned binding means the element represents an internal node
//   with no meaningful concept of a direction
sealed trait UndirectionedBinding extends Binding { def direction = None }

// This is the default binding, represents data not yet positioned in the graph
case class UnboundBinding(direction: Option[Direction])
    extends Binding with UnconstrainedBinding


// A synthesizable binding is 'bound into' the hardware graph
object SynthesizableBinding {
  def unapply(target: Binding): Boolean = target.isInstanceOf[SynthesizableBinding]
  // Type check OK because Binding and SynthesizableBinding is sealed
}
sealed trait SynthesizableBinding extends Binding
case class LitBinding() // will eventually have literal info
    extends SynthesizableBinding with UnconstrainedBinding with UndirectionedBinding

case class MemoryPortBinding(enclosure: UserModule)
    extends SynthesizableBinding with ConstrainedBinding with UndirectionedBinding

// TODO(twigg): Ops between unenclosed nodes can also be unenclosed
// However, Chisel currently binds all op results to a module
case class OpBinding(enclosure: UserModule)
    extends SynthesizableBinding with ConstrainedBinding with UndirectionedBinding

case class PortBinding(enclosure: BaseModule, direction: Option[Direction])
    extends SynthesizableBinding with ConstrainedBinding

case class RegBinding(enclosure: UserModule)
    extends SynthesizableBinding with ConstrainedBinding with UndirectionedBinding

case class WireBinding(enclosure: UserModule)
    extends SynthesizableBinding with ConstrainedBinding with UndirectionedBinding
