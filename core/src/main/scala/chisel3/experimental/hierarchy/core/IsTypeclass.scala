package chisel3.experimental.hierarchy.core

trait Hierarchicalizer[-A] {
  def hierarchy(context: Hierarchy[A]): Option[Proxy[IsHierarchical]]
  def asUnderlying[X <: A](value: X): Proxy[X]
}

trait IsTypeclass[B] {
  type C
  def apply[A](value: B, hierarchy: Hierarchy[A])(implicit h: Hierarchicalizer[A]): C
}
