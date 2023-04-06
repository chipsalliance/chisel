// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.BaseModule
import chisel3.internal.{HasId, PseudoModule}
import chisel3.internal.firrtl.{Component, Ref}

/** Represents a Definition root module, when accessing something from a definition
  *
  * @note This is necessary to distinguish between the toTarget behavior for a Module returned from a Definition,
  * versus a normal Module. A normal Module.toTarget will always return a local target. If calling toTarget
  * on a Module returned from a Definition (and thus wrapped in an Instance), we need to return the non-local
  * target whose root is the Definition. This DefinitionClone is used to represent the root parent of the
  * InstanceClone (which represents the returned module).
  */
private[chisel3] class DefinitionClone[T <: BaseModule](val getProto: T) extends PseudoModule with core.IsClone[T] {
  override def toString = s"DefinitionClone(${getProto})"
  override private[chisel3] def _definitionIdentifier = getProto.definitionIdentifier
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Do not call default addId function, which may modify a module that is already "closed"
  override def addId(d: HasId): Unit = ()
  // Necessary for toTarget to work
  private[chisel3] def initializeInParent(): Unit = ()
  // Module name is the same as proto's module name
  override def desiredName: String = getProto.name
}
