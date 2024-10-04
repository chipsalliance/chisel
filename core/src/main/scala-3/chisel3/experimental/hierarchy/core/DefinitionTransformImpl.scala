package chisel3.experimental.hierarchy.core

import chisel3.experimental.BaseModule

object DefinitionTransformImpl {
  def _applyImpl[T <: BaseModule with IsInstantiable](proto: => T): Definition[T] = Definition.do_apply(proto)
}
