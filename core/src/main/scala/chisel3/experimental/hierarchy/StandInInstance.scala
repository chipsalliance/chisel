package chisel3.experimental.hierarchy
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.internal.PseudoModule
import chisel3.internal.firrtl._
import chisel3._


/** Represents a module viewed from a different instance context.
  *
  * @note Why do we need both experimental.hierarchy.StandInModule and InstanceClone? If we are annotating a reference in a module-clone,
  * all submodules must be also be 'cloned' so the toTarget can be computed properly. However, we don't need separate
  * connectable ports for this instance; all that's different from the proto is the parent.
  *
  * @note In addition, the instance name of an InstanceClone is going to be the SAME as the proto, but this is not true
  * for experimental.hierarchy.StandInModule.
  */
private[chisel3] final class StandInInstance[T <: BaseModule](val getProto: T, val instName: () => String, parent: Option[IsHierarchicable]) extends PseudoModule with IsStandIn[T] {
  _parent = parent match {
    case b: BaseModule => Some(b)
    case other => None
  }

  // ======== THINGS TO MAKE CHISEL WORK ========

  override def toString = s"InstanceClone(${getProto})"
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Necessary for toTarget to work
  private[chisel3] def setAsInstanceRef(): Unit = { this.setRef(Ref(instName())) }
  // This module doesn't acutally exist in the FIRRTL so no initialization to do
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
  // Instance name is the same as proto's instance name
  override def instanceName = instName()
  // Module name is the same as proto's module name
  override def desiredName: String = getProto.name
}
