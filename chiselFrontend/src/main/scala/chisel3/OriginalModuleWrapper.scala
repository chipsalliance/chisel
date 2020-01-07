// See LICENSE for license details.

package chisel3

import chisel3.SpecifiedDirection.Unspecified
import chisel3.experimental.{BaseModule, CloneModuleAsRecord, ExtModule}
import chisel3.internal.BaseModule.ClonePorts
import chisel3.internal.Builder.pushCommand
import chisel3.internal._
import chisel3.internal.firrtl.{Component, DefBlackBox, DefInstance, DefInvalid, ModuleIO, Port}
import chisel3.internal.sourceinfo.{ImportTransform, InstTransform, SourceInfo}
import _root_.firrtl.annotations.{CompleteTarget, IsMember}
import chisel3.incremental.{ItemTag, Stash}

import scala.collection.immutable.ListMap
import scala.language.experimental.macros

private class OriginalModuleWrapper[T <: BaseModule] private[chisel3](moduleName: String,
                                                                      ports: Record
                                                                     )(implicit sourceInfo: SourceInfo,
                                                                        compileOptions: CompileOptions) extends BlackBox {
  override def desiredName: String = moduleName

  val io = IO(ports)

}



class InstanceHandle[T <: BaseModule] private[chisel3](tag: ItemTag[T],
                                                       packge: Option[String]
                                                       //version: Option[(String, Option[(Int, Option[(Int, Option[Int])])])],
                                                      )(implicit sourceInfo: SourceInfo,
                                                        compileOptions: CompileOptions) /*extends HasId*/ {


  private[chisel3] val wrapper = if(Builder.stash.get.retrieve(tag, packge).nonEmpty) {
    val importedModule = Builder.stash.get.retrieve(tag, packge).get
    val record = new ClonePorts(importedModule.getModulePorts:_*)
    Module.do_apply(new OriginalModuleWrapper[T](importedModule.name, record))
  } else {
    throwException(s"Cannot find imported module with tag $tag in package $packge")
  }

  private def set(): Unit = { }

  private def unset(): Unit = { }

  def apply[X](f: T => X): X = {
    set()
    val importedModule = Builder.stash.get.retrieve(tag, packge).get
    val portMap = importedModule.getModulePorts.zip(wrapper.io.elements.values).toMap
    val ret = f(importedModule)
    unset()
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
  /*
  override def toNamed= ???
  override def toTarget: IsMember = ???
  override def toAbsoluteTarget: IsMember = ???
   */
}

object InstanceHandle extends SourceInfoDoc {
  def apply[T <: BaseModule](tag: ItemTag[T], packge: Option[String]): InstanceHandle[T] = macro ImportTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule](tag: ItemTag[T], packge: Option[String])
                               (implicit sourceInfo: SourceInfo,
                                compileOptions: CompileOptions): InstanceHandle[T] = {
    new InstanceHandle(tag, packge)
  }

  /** Returns the implicit Clock */
  def clock: Clock = Builder.forcedClock
  /** Returns the implicit Reset */
  def reset: Reset = Builder.forcedReset
  /** Returns the current Module */
  def currentModule: Option[BaseModule] = Builder.currentModule
}
