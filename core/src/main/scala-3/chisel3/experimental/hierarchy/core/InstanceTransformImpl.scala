package chisel3.experimental.hierarchy.core

import chisel3.experimental.BaseModule

object InstanceTransformImpl {
  def _applyImpl[T <: BaseModule with IsInstantiable](definition: Definition[T]): Instance[T] = Instance.do_apply(definition)
}
