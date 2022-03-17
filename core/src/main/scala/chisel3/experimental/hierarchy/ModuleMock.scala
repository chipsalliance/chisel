// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.PseudoModule
import chisel3.internal.firrtl._
import chisel3._

/** Proxy of Instance when viewed from a different Hierarchy
  *
  * Represents a non-local instance.
  *
  * @param genesis Proxy of the same proto with a less-specific hierarchical path
  * @param contexts contains contextual values when viewed from this proxy
  */
private[chisel3] final case class ModuleMock[T <: BaseModule] private (
  val genesis:  InstanceProxy[T] with BaseModule,
  val contexts: Seq[Context[T]])
    extends PseudoModule
    with Mock[T] {
  
  override def lineage = _parent.get.asInstanceOf[Proxy[BaseModule]]

  // ======== THINGS TO MAKE CHISEL WORK ========

  override def toString = s"ModuleMock(${proto})"
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Necessary for toTarget to work
  private[chisel3] def setAsInstanceRef(): Unit = { this.setRef(Ref(localProxy.asInstanceOf[BaseModule].instanceName)) }
  // This module doesn't acutally exist in the FIRRTL so no initialization to do
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
  // Module name is the same as proto's module name
  override def desiredName:  String = proto.name
  override def instanceName: String = localProxy.asInstanceOf[BaseModule].instanceName
}

private[chisel3] object ModuleMock {
  def apply[T <: BaseModule](
    genesis:  InstanceProxy[T] with BaseModule,
    lineage:  BaseModule,
    contexts: Seq[Context[T]]
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): ModuleMock[T] = {
    val x = Module.do_pseudo_apply(new ModuleMock(genesis, contexts))
    x._parent = Some(lineage)
    x
  }
}
