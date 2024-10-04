package chisel3.experimental.hierarchy.core

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.DefinitionTransform

object DefinitionTransformImpl {
  def _applyImpl[T <: BaseModule with IsInstantiable](proto: => T): Definition[T] = macro DefinitionTransform.apply[T]
}
