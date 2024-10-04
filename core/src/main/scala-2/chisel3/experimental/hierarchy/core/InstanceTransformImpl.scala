package chisel3.experimental.hierarchy.core

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.InstanceTransform

object InstanceTransformImpl {
  def _applyImpl[T <: BaseModule with IsInstantiable](definition: Definition[T]): Instance[T] = macro InstanceTransform.apply[T]
}
