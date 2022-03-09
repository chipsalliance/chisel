package chisel3.experimental.hierarchy
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.PseudoModule
import chisel3.internal.firrtl._
import chisel3._


/** Represents a module viewed from a different instance context.
  *
  * @note Why do we need both experimental.hierarchy.ModuleClone and InstanceClone? If we are annotating a reference in a module-clone,
  * all submodules must be also be 'cloned' so the toTarget can be computed properly. However, we don't need separate
  * connectable ports for this instance; all that's different from the proto is the parent.
  *
  * @note In addition, the instance name of an InstanceClone is going to be the SAME as the proto, but this is not true
  * for experimental.hierarchy.ModuleClone.
  */
private[chisel3] final case class ModuleMock[T <: BaseModule] private (
    val genesis: HierarchicalProxy[T, BaseModule] with BaseModule
) extends PseudoModule with Mock[T, BaseModule] {
  def parent = _parent.get

  // ======== THINGS TO MAKE CHISEL WORK ========

  override def toString = s"ModuleMock(${proto})"
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Necessary for toTarget to work
  private[chisel3] def setAsInstanceRef(): Unit = { this.setRef(Ref(local.asInstanceOf[BaseModule].instanceName)) }
  // This module doesn't acutally exist in the FIRRTL so no initialization to do
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
  // Module name is the same as proto's module name
  override def desiredName: String = proto.name
  override def instanceName: String = local.asInstanceOf[BaseModule].instanceName
}

private[chisel3] object ModuleMock {
  def apply[T <: BaseModule](
    genesis: HierarchicalProxy[T, BaseModule] with BaseModule,
    lineage: BaseModule
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
  ): ModuleMock[T] = {
    val x = Module.do_pseudo_apply(new ModuleMock(genesis))
    x._parent = Some(lineage)
    x
  }
}