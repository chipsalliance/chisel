// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable

// ==========================================
// User-facing Traits
// ==========================================

// Context independent value. Use for immutable case classes which are true for all Hierarchy[_]
trait IsLookupable

// Context dependent value with a clone method, so clones itself to the new context
// TODO: Right now this is not used nor implemented
trait IsContextual

// Context dependent that lacks a clone method, and so wraps itself in Hierarchy[_] to represent the context-specific value
trait IsInstantiable

// ==========================================
// Library-facing Traits, NOT FOR USERS
// Additional library-facing traits are not in this file
// ==========================================

// Implemented by a library so we can create a Definition of an underlying Proto
trait ProxyDefiner[P] {
  def apply(f: => P): DefinitionProxy[P]
}

// Implemented by a library so we can create an Instance of a Definition
trait ProxyInstancer[P] {
  def apply(definition: Definition[P], contexts: Seq[RootContext[P]]): Clone[P]
}

trait ContextualInstancer[V, P] {
  def apply(value: V): Contextual[V, P]
}
