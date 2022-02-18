package chisel3.experimental.hierarchy.core

trait IsTypeclass[B] {
  type C
  def apply[A](value: B, hierarchy: Hierarchy[A]): C
}