package chisel3.experimental.hierarchy.core

// ==========================================
// User-facing Traits
// ==========================================

// Context independent value. Use for immutable case classes which are true for all Hierarchy[_]
trait IsLookupable

// Context dependent value with a clone method, so clones itself to the new context
trait IsContextual

// Context dependent that lacks a clone method, and so wraps itself in Hierarchy[_] to represent the context-specific value
trait IsInstantiable

// Context dependent that lacks a clone method, and also creates a new context for values looked up from it
trait IsContext


// ==========================================
// Library-facing Traits, NOT FOR USERS
// Additional library-facing traits are not in this file
// ==========================================

// Implemented by a library so we can create a Definition of an underlying Proto
trait ProxyDefiner[P] {
  def apply(f: => P): Proxy[P]
}

// Implemented by a library so we can create an Instance of a Definition
trait ProxyInstancer[P] {
  def apply(definition: Definition[P]): Proxy[P]
}

// Use for a library to have a standin for an IsContext proto
//   This is the thing that interacts with a library's internal representations and state properly to
//   manage context
trait ContextStandIn[+P <: IsContext] extends IsStandIn[P] {
  def asProxy: Proxy[P] = StandIn(this)
}
