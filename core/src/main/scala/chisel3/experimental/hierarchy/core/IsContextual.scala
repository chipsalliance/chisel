package chisel3.experimental.hierarchy.core

// Marker Trait
trait IsContextual

// Wrapper Class
final case class Contextual[V](value: V)

// No Underlying Classes needed; Contextuals clone themselves into their new hierarchy

// Typeclass Trait
trait Contextualizer[V] extends IsTypeclass[V] {
  type R
  def apply[H](value: V, hierarchy: Hierarchy[H]): R
}

// Default Typeclass Implementations
object Contextualizer {
  implicit def isLookupable[L <: IsLookupable] = new Contextualizer[L] {
    type R = L
    def apply[H](v: L, hierarchy: Hierarchy[H]): L = v
  }
  implicit def isContextual[V] = new Contextualizer[Contextual[V]] {
    type R = V
    def apply[H](v: Contextual[V], hierarchy: Hierarchy[H]): V = v.value
  }
  implicit def isOther[X](implicit proxifier: Proxifier[X]) = new Contextualizer[X] {
    type R = proxifier.R
    def apply[H](v: X, hierarchy: Hierarchy[H]): R = proxifier(v, hierarchy)
  }
}


//TODO: will need to handle nested Contextuals with another typeclass