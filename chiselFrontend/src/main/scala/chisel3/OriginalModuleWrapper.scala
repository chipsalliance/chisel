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
import chisel3.incremental.{ItemTag, Stash}

import scala.collection.immutable.ListMap
import scala.language.experimental.macros

private class OriginalModuleWrapper[T <: BaseModule] private[chisel3](instanceName: String,
                                                                      accessImport: () => (String, ClonePorts)
                                                                     )(implicit sourceInfo: SourceInfo,
                                                                        compileOptions: CompileOptions) extends ExtModule {
  private lazy val (moduleName, ports) = accessImport()
  //override def desiredName = moduleName
  forceName(instanceName, _namespace)

  private[chisel3] override def generateComponent(): Component = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    val firrtlPorts = ports.elements.toSeq map {
      case (name, port) => Port(port, port.specifiedDirection)
    }
    val component = DefBlackBox(this, name, firrtlPorts, SpecifiedDirection.Unspecified, params)

    _component = Some(component)
    component
  }

}



class InstanceHandle[T <: BaseModule] private[chisel3](instanceName: String,
                                                       tag: ItemTag[T],
                                                       packge: Option[String]
                                                      )(implicit sourceInfo: SourceInfo,
                                                        compileOptions: CompileOptions) {
  val stash = Builder.stash.get
  if(Builder.stash.get.retrieve(tag, packge).nonEmpty) {
    val accessImport = () => {
      val importedModule = Builder.stash.get.retrieve(tag, packge).get
      (importedModule.name, new ClonePorts(importedModule.getModulePorts:_*))
    }
    Module.do_apply(new OriginalModuleWrapper[T](instanceName, accessImport))
  } else {
    throwException(s"Cannot find imported module with tag $tag in package $packge")
  }

  private def set(): Unit = { }

  private def unset(): Unit = { }

  def apply[X](f: T => X): X = {
    set()
    val ret = f(Builder.stash.get.retrieve(tag, packge).get)
    unset()
    ret match {
      case d: Data =>
        println("Returning data")
        ret
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
  def apply[T <: BaseModule](name: String, tag: ItemTag[T], packge: Option[String]): InstanceHandle[T] = macro ImportTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule](name: String, tag: ItemTag[T], packge: Option[String])
                               (implicit sourceInfo: SourceInfo,
                                compileOptions: CompileOptions): InstanceHandle[T] = {
    new InstanceHandle(name, tag, packge)
  }

  /** Returns the implicit Clock */
  def clock: Clock = Builder.forcedClock
  /** Returns the implicit Reset */
  def reset: Reset = Builder.forcedReset
  /** Returns the current Module */
  def currentModule: Option[BaseModule] = Builder.currentModule
}
