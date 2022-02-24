package chisel3.experimental.hierarchy.core

trait Contexter[C] {
  def lookupContext[H](hierarchy: Hierarchy[H]): Option[C]
}

trait IsTypeclass[V] {
  type R
  def apply[H](value: V, hierarchy: Hierarchy[H]): R
}
