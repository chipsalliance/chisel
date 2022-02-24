package chisel3.experimental.hierarchy.core

// Wrapper Class
final case class Contextual[V](value: V)

// Typeclass Trait
trait Contextualizer[V]  {
  type R
  def apply[P](value: V, hierarchy: Hierarchy[P]): R
}

// Default Typeclass Implementations
object Contextualizer {
  implicit def isLookupable[L <: IsLookupable] = new Contextualizer[L] {
    type R = L
    def apply[P](v: L, hierarchy: Hierarchy[P]): L = v
  }
  //implicit def isContextual[V] = new Contextualizer[Contextual[V]] {
  //  type R = V
  //  def apply[P](v: Contextual[V], hierarchy: Hierarchy[P]): V = v.value
  //}
  //implicit def isOther[X](implicit proxifier: Proxifier[X]) = new Contextualizer[X] {
  //  type R = proxifier.R
  //  def apply[P](v: X, hierarchy: Hierarchy[P]): R = proxifier(v, hierarchy)
  //}
}
