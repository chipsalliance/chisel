// See LICENSE for license details.

package chisel3

import chisel3.experimental.BaseModule
import chisel3.internal.BaseModule.ClonePorts
import chisel3.internal._
import chisel3.internal.sourceinfo.SourceInfo

import scala.language.experimental.macros

private class OriginalModuleWrapper[T <: BaseModule] private[chisel3](
    moduleName: String,
    ports: Record
)( implicit sourceInfo: SourceInfo, compileOptions: CompileOptions ) extends BlackBox {
  override def desiredName: String = moduleName
  val io = IO(ports)
}

class InstanceHandle[T <: BaseModule] private[chisel3](
    packge: Option[GeneratorPackage[BaseModule]],
    instance: T,
    index: Int
)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {

  private[chisel3] val wrapper = {
    val record = new ClonePorts(instance.getModulePorts:_*)
    Module.do_apply(new OriginalModuleWrapper[T](packge.map(_.phase.name + "_").getOrElse("") + instance.name, record))
  }

  val portMap = instance.getModulePorts.zip(wrapper.io.elements.values).toMap

  def apply[X](f: T => X): X = {
    val ret = f(instance)
    ret match {
      case d: Data if !portMap.contains(d) => d.cloneType.asInstanceOf[X]
      case d: Data => portMap(d).asInstanceOf[X]
      case _ => ret
    }
  }

  /*
  def toTarget(f: T => CompleteTarget): CompleteTarget = {
    import _root_.firrtl.annotations._
    val circuit = originalModuleWrapper.toTarget.circuit
    f(Stash.module(originalId).asInstanceOf[T]) match {
      case ct: CircuitTarget   => ct.copy(circuit = circuit)
      case mt: ModuleTarget    => mt.copy(circuit = circuit)
      case it: InstanceTarget  => it.copy(circuit = circuit)
      case rt: ReferenceTarget => rt.copy(circuit = circuit)
    }
  }

  def toAbsoluteTarget(f: T => CompleteTarget): CompleteTarget = {
    import _root_.firrtl.annotations._
    val path = originalModuleWrapper.toAbsoluteTarget
    val t = toTarget(f)
    t match {
      case mt: IsMember => mt.setPathTarget(path)
      case other => other
    }
  }
   */
}

object InstanceHandle extends SourceInfoDoc {
  //def apply[T <: BaseModule](tag: Class[T], packge: Option[String]): InstanceHandle[T] = macro ImportTransform.apply[T]

  ///** @group SourceInfoTransformMacro */
  //def do_apply[T <: BaseModule](tag: Class[T], packge: Option[String])
  //                             (implicit sourceInfo: SourceInfo,
  //                              compileOptions: CompileOptions): InstanceHandle[T] = {
  //  new InstanceHandle(tag, packge)
  //}

  /** Returns the implicit Clock */
  //def clock: Clock = Builder.forcedClock
  /** Returns the implicit Reset */
  //def reset: Reset = Builder.forcedReset
  /** Returns the current Module */
  //def currentModule: Option[BaseModule] = Builder.currentModule
}
