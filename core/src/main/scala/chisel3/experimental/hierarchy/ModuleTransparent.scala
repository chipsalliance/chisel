// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.hierarchy.core._
import chisel3.internal.firrtl._
import chisel3.internal.{throwException, HasId, Namespace, PseudoModule}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import chisel3._
import Utils._

/** Proxy of Instance which was created by calling .toInstance from a module
  *
  * @param suffixProxy Proxy of the Module Definition of the proto (original module)
  */
private[chisel3] final class ModuleTransparent[T <: BaseModule] private (
  val suffixProxy: ModuleRoot[T])
    extends PseudoModule
    with Transparent[T] {

  lazy val ioMap: Map[Data, Data] = proto.getChiselPorts.map { case (_, data) => data -> data }.toMap
  contextuals ++= suffixProxy.contextuals
  def debug = getTarget.toString

  // ======== THINGS TO MAKE CHISEL WORK ========

  override def toString = s"ModuleTransparent(${proto})"
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Necessary for toTarget to work
  private[chisel3] def setAsInstanceRef(): Unit = { this.setRef(Ref(instanceName)) }
  // This module doesn't acutally exist in the FIRRTL so no initialization to do
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
  // Instance name is the same as proto's instance name
  override def instanceName = proto.instanceName
  // Module name is the same as proto's module name
  override def desiredName: String = proto.name
}

private[chisel3] object ModuleTransparent {
  def apply[T <: BaseModule](
    suffixProxy: ModuleRoot[T]
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): ModuleTransparent[T] = {
    if(suffixProxy.transparentProxy.nonEmpty) suffixProxy.transparentProxy.get.asInstanceOf[ModuleTransparent[T]] else {
      val ret = Module.do_pseudo_apply(new ModuleTransparent(suffixProxy))
      ret._parent = suffixProxy.proto._parent
      suffixProxy.transparentProxy = Some(ret)
      ret
    }
  }
}
