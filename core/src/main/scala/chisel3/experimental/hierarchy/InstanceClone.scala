// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.BaseModule
import chisel3.internal.PseudoModule
import chisel3.internal.firrtl.{Component, Ref}

/** Represents a module viewed from a different instance context.
  *
  * @note Why do we need both ModuleClone and InstanceClone? If we are annotating a reference in a module-clone,
  * all submodules must be also be 'cloned' so the toTarget can be computed properly. However, we don't need separate
  * connectable ports for this instance; all that's different from the proto is the parent.
  *
  * @note In addition, the instance name of an InstanceClone is going to be the SAME as the proto, but this is not true
  * for ModuleClone.
  */
private[chisel3] final class InstanceClone[T <: BaseModule](val getProto: T, val instName: () => String)
    extends PseudoModule
    with core.IsClone[T] {
  override def toString = s"InstanceClone(${getProto})"
  override private[chisel3] def _definitionIdentifier = getProto.definitionIdentifier
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Necessary for toTarget to work
  private[chisel3] def setAsInstanceRef(): Unit = { this.setRef(Ref(instName())) }
  // This module doesn't acutally exist in the FIRRTL so no initialization to do
  private[chisel3] def initializeInParent(): Unit = ()
  // Instance name is the same as proto's instance name
  override def instanceName = instName()
  // Module name is the same as proto's module name
  override def desiredName: String = getProto.name
}
