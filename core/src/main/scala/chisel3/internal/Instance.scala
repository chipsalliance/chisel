package chisel3.internal

import chisel3.{BlackBox, CompileOptions, Record}
import chisel3.experimental.BaseModule
import chisel3.internal.BaseModule.ClonePorts
import chisel3.internal.sourceinfo.SourceInfo

import scala.collection.mutable

case class InstanceKey(name: String, parentModule: Long, module: Long)

object Instance {
  def apply[T <: BaseModule](module: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, valName: ValName): T = {
    createInstance(module, Some(valName.name))
  }
  def createInstance[T <: BaseModule](module: T, name: Option[String])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    require(name.isDefined)
    require(Builder.currentModule.isDefined)
    val key = InstanceKey(name.get, Builder.currentModule.get._id, module._id)
    val wrapper = {
      val record = new ClonePorts(module.getModulePorts:_*)
      chisel3.Module.do_apply(new Instance[T](module.name, record, module)).suggestName(name.get)
    }
    Builder.addInstance(key, wrapper)
    module
  }
}

class Instance[T <: BaseModule] private[chisel3]( moduleName: String, ports: Record, module: T )
                                                ( implicit sourceInfo: SourceInfo,
                                                  compileOptions: CompileOptions ) extends BlackBox {
  override def desiredName: String = moduleName
  val io = IO(ports)
  val portMap = module.getModulePorts.zip(io.elements.values).toMap
}


