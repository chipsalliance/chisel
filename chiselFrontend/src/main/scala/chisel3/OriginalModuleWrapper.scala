// See LICENSE for license details.

package chisel3

import chisel3.SpecifiedDirection.Unspecified
import chisel3.experimental.{BaseModule, CloneModuleAsRecord, ExtModule}
import chisel3.internal.BaseModule.ClonePorts
import chisel3.internal.Builder.pushCommand
import chisel3.internal._
import chisel3.internal.firrtl.{Component, DefBlackBox, DefInstance, DefInvalid, ModuleIO, Port}
import chisel3.internal.sourceinfo.{ImportTransform, InstTransform, SourceInfo}
import _root_.firrtl.annotations.CompleteTarget
import chisel3.incremental.Stash

import scala.collection.immutable.ListMap
import scala.language.experimental.macros

private class OriginalModuleWrapper[T <: BaseModule] private[chisel3](id: Long, instanceName: String)(implicit sourceInfo: SourceInfo,
                                                                        compileOptions: CompileOptions) extends ExtModule {
  private def module: T = Stash.module(id).asInstanceOf[T]

  override def desiredName = module.name
  forceName(instanceName, _namespace)

  Stash.setParent(id, this._parent.get._id)
  Stash.setLink(id, this._id)


  private[chisel3] override def generateComponent(): Component = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    val record = new ClonePorts(module.getModulePorts:_*)
    val firrtlPorts = record.elements.toSeq map {
      case (name, port) => Port(port, port.specifiedDirection)
    }
    val component = DefBlackBox(this, name, firrtlPorts, SpecifiedDirection.Unspecified, params)

    _component = Some(component)
    component
  }

}

class InstanceHandle[T <: BaseModule] private[chisel3](originalId: Long,
                                                       originalModuleWrapper: OriginalModuleWrapper[T]) {
  private def set(): Unit = {
    Stash.setActiveParent(originalId, originalModuleWrapper._parent.get._id)
    Stash.setActiveLink(originalId, originalModuleWrapper._id)
  }

  private def unset(): Unit = {
    Stash.clearActiveParent(originalId)
    Stash.clearActiveLink(originalId)
  }

  def apply[X](f: T => X): X = {
    set()
    val ret = f(Stash.module(originalId).asInstanceOf[T])
    unset()
    ret
  }

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
}

object InstanceHandle extends SourceInfoDoc {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param id the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule](name: String, id: Long): InstanceHandle[T] = macro ImportTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule](name: String, id: Long)
                               (implicit sourceInfo: SourceInfo,
                                compileOptions: CompileOptions): InstanceHandle[T] = {
    val module: OriginalModuleWrapper[T] = Module.do_apply(new OriginalModuleWrapper[T](id, name))  // bc is actually evaluated here
    new InstanceHandle(id, module)
  }
  def apply[T <: BaseModule](name: String, id: T): InstanceHandle[T] = macro ImportTransform.apply[T]
  def do_apply[T <: BaseModule](name: String, id: T)
                               (implicit sourceInfo: SourceInfo,
                                compileOptions: CompileOptions): InstanceHandle[T] = do_apply(name, id._id)

  /** Returns the implicit Clock */
  def clock: Clock = Builder.forcedClock
  /** Returns the implicit Reset */
  def reset: Reset = Builder.forcedReset
  /** Returns the current Module */
  def currentModule: Option[BaseModule] = Builder.currentModule
}
