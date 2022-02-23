package chisel3.experimental.hierarchy.core

trait Hierarchicable[-A] {
  def hierarchy(context: Hierarchy[A]): Option[Underlying[IsHierarchicable]]
  def asUnderlying[X <: A](value: X): Underlying[X]
}

trait IsTypeclass[B] {
  type C
  def apply[A](value: B, hierarchy: Hierarchy[A])(implicit h: Hierarchicable[A]): C
}
