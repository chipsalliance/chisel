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
// Library-facing Trait, NOT FOR USERS
// ==========================================

// Use for a library to have a standin for a proto, which can be the thing that updates the library's
//  internal representations and state properly
trait IsStandIn[+T] {

  def parent: Option[IsContext]
  def proto: T

  def toInstance:   Instance[T]
  def toDefinition: Definition[T]
  /** Determines whether another object is a clone of the same proxy proto
    *
    * @param a
    */
  def hasSameProto(a: Any): Boolean = {
    val aProto = a match {
      case is: IsStandIn[_] => is.proto
      case other => other
    }
    this == aProto || proto == aProto
  }
}
