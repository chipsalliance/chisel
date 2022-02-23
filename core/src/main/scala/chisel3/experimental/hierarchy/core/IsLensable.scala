package chisel3.experimental.hierarchy.core

/*
// Marker Trait
trait IsLensable

// Wrapper Class
final case class Lense[T](proxy: Proxy[T, IsHierarchical])

// Underlying Classes; For now, just use IsHierarchical's until proven otherwise

// Typeclass Trait
trait Lensify[B] extends IsTypeclass[B] {
  type C
  def apply[A](b: B, context: Lense[A]): C
}

object Lensify {
  // I believe lensing on isLookupable should be illegal; don't define this?!?
  implicit def isLookupable[L <: IsLookupable] = new Lensify[L] {
    type C = L
    def apply[C](b: L, context: Lense[C]): L = ??? //ERROR!! Cannot do this!!
  }
  implicit def IsHierarchical[I <: IsHierarchical] = new Lensify[Hierarchy[I]] {
    type C = Lense[I]
    def apply[C](b: Hierarchy[I], context: Lense[C]): Lense[I] = Lense(b.proxy)
  }
}
*/



//TODO: will need to handle lensing nested Contextuals with another typeclass
// TODO: nested contextuals
//final case class ContextualLense[T, V](value: V, parent: Proxy[T, IsHierarchical])
  // TODO: nested contextuals
  //implicit def isContextual[I <: IsContextual] = new Lensify[Contextual[I], Edit[I]] {
  //  def lensify[C](b: Contextual[I], context: Lense[C]): Edit[I] = Edit(b.value)
  //}