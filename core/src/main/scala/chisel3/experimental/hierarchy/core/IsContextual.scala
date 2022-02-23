package chisel3.experimental.hierarchy.core

// Marker Trait
trait IsContextual

// Wrapper Class
final case class Contextual[V](value: V)

// No Underlying Classes needed; Contextuals clone themselves into their new hierarchy

// Typeclass Trait
trait Contextualizer[B] extends IsTypeclass[B] {
  type C
  def apply[A](value: B, hierarchy: Hierarchy[A])(implicit h: Hierarchicalizer[A]): C
}

// Default Typeclass Implementations
object Contextualizer {
  implicit def isLookupable[L <: IsLookupable] = new Contextualizer[L] {
    type C = L
    def apply[A](v: L, hierarchy: Hierarchy[A])(implicit h: Hierarchicalizer[A]): L = v
  }
  implicit def isContextual[V] = new Contextualizer[Contextual[V]] {
    type C = V
    def apply[A](v: Contextual[V], hierarchy: Hierarchy[A])(implicit h: Hierarchicalizer[A]): V = v.value
  }
  implicit def isOther[X](implicit proxifier: Proxifier[X]) = new Contextualizer[X] {
    type C = proxifier.C
    def apply[A](v: X, hierarchy: Hierarchy[A])(implicit h: Hierarchicalizer[A]): C = proxifier(v, hierarchy)
  }
}


//TODO: will need to handle nested Contextuals with another typeclass