package chisel3.libs.aspect

import chisel3.Data
import chisel3.experimental.MultiIOModule
import chisel3.internal.HasId

abstract class Snippet[M<:MultiIOModule, D<:HasId] {
  implicit def ref2cmSource[T<:Data](ref: T): CrossModuleSource[T] = CrossModuleSource(ref)
  implicit def ref2cmSink[T<:Data](ref: T): CrossModuleSink[T] = CrossModuleSink(ref)

  case class CrossModuleSource[T<:Data](cmr: T) {
    def ref: T = Aspect.dynamicContextVar.value.get.addInput(cmr)
    def i: T = ref
  }
  case class CrossModuleSink[T<:Data](cmr: T) {
    def ref: T = Aspect.dynamicContextVar.value.get.addOutput(cmr)
    def o: T = ref
  }

  def snip(top: M): D
}
