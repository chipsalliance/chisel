package chisel3.experimental.hierarchy.core

// Marker Trait
trait IsContextual

// Wrapper Class
final case class Contextual[V](value: V)

// No Underlying Classes needed; Contextuals clone themselves into their new hierarchy

// Typeclass Trait
trait Contextualize[B] extends IsTypeclass[B] {
  type C
  def apply[A](value: B, hierarchy: Hierarchy[A])(implicit h: Hierarchicable[A]): C
}

// Default Typeclass Implementations
object Contextualize {
  implicit def isLookupable[L <: IsLookupable] = new Contextualize[L] {
    type C = L
    def apply[A](v: L, hierarchy: Hierarchy[A])(implicit h: Hierarchicable[A]): L = v
  }
  implicit def isContextual[V] = new Contextualize[Contextual[V]] {
    type C = V
    def apply[A](v: Contextual[V], hierarchy: Hierarchy[A])(implicit h: Hierarchicable[A]): V = v.value
  }
  implicit def isOther[X](implicit viewable: Viewable[X]) = new Contextualize[X] {
    type C = viewable.C
    def apply[A](v: X, hierarchy: Hierarchy[A])(implicit h: Hierarchicable[A]): C = viewable(v, hierarchy)
  }
}


//TODO: will need to handle nested Contextuals with another typeclass