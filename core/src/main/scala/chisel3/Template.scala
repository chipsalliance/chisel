package chisel3

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.SourceInfo

case class Template[T <: BaseModule](
    instance: T,
    packge: Option[GeneratorPackage[BaseModule]]
) {
  var index = 0
  def instantiate()(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): InstanceHandle[T] = {
    val newInstanceHandle = new InstanceHandle(packge, instance, index)
    index += 1
    newInstanceHandle
  }
}
