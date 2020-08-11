package chisel3.internal

import java.lang.reflect.Field

import chisel3.{BlackBox, CompileOptions, Record}
import chisel3.experimental.BaseModule
import chisel3.internal.BaseModule.ClonePorts
import chisel3.internal.sourceinfo.SourceInfo

import scala.collection.mutable
import scala.runtime.{IntRef, ObjectRef}

case class InstanceKey(name: String, parentModule: Long, module: Long)

object Instance {
  def apply[T <: BaseModule](module: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, valName: ValName): T = {
    createInstance(module, Some(valName.name))
  }
  def createInstance[T <: BaseModule](module: T, name: Option[String])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    require(name.isDefined)
    val wrapper = {
      val record = new ClonePorts(module.getModulePorts:_*)
      chisel3.Module.do_apply(new Instance[T](module.name, record, module)).suggestName(name.get)
    }
    val constr = module.getClass().getConstructors.head

    // Special case for inner classes - we must find the outer class and pass it to the constructor of our nullified
    //  inner class
    val paramVals: Seq[Object] = try {
      val fields = module.getClass.getDeclaredFields.collectFirst {
        case f: Field if f.toString.contains("$outer") => f
      }.toList
      fields.foreach(_.setAccessible(true))
      val outers: Seq[Object] = fields.map(f => f.get(module))
      val otherPs: Seq[Object] = constr.getParameterTypes.drop(outers.size).map {
        // TODO: Support more than just int primitives
        case c: Class[_] if c.isPrimitive => 0.asInstanceOf[Object]
        case _ => null.asInstanceOf[Object]
      }
      outers ++ otherPs
    } catch { case e: NoSuchFieldException =>
      constr.getParameterTypes.map {
        case c: Class[_] if c.isPrimitive => 0.asInstanceOf[Object]
        case _ => null.asInstanceOf[Object]
      }
    }
    require(Builder.currentModule.isDefined)
    Builder.setBackingModule(module, wrapper)
    val fakeModule = chisel3.Module.do_apply(constr.newInstance(paramVals:_*).asInstanceOf[T])
    Builder.clearBackingModule()
    fakeModule
  }
}

class Instance[T <: BaseModule] private[chisel3]( moduleName: String, ports: Record, module: T )
                                                ( implicit sourceInfo: SourceInfo,
                                                  compileOptions: CompileOptions ) extends BlackBox {
  override def desiredName: String = moduleName
  val io = IO(ports)
  val portMap = module.getModulePorts.zip(io.elements.values).toMap
}


