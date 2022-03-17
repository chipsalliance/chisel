// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.language.experimental.macros
import chisel3._
import chisel3.internal.sourceinfo.SourceInfo

import chisel3.internal.{HasId, PseudoModule}
import chisel3.internal.firrtl._
import chisel3.experimental.hierarchy.core._
import scala.collection.mutable.HashMap
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import firrtl.annotations.{IsModule, ModuleTarget}

/** Proxy of Definition of a user-defined module.
  *
  * @param proto Underlying module which this is the definition of
  */
private[chisel3] final case class ModuleDefinition[T <: BaseModule](proto: T)
    extends PseudoModule
    with DefinitionProxy[T] {
  override def equals(a: Any): Boolean = {
    a match {
      case d: ModuleDefinition[_] if d.proto == proto && d._circuit == _circuit => true
      case _ => false
    }
  }

  def toInstance: core.Instance[T] = ???

  // ======== THINGS TO MAKE CHISEL WORK ========

  //override def toString = s"ModuleDefinition($proto, $circuit)"
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Do not call default addId function, which may modify a module that is already "closed"
  override def addId(d: HasId): Unit = ()
  // Necessary for toTarget to work
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
  // Module name is the same as proto's module name
  override def desiredName: String = proto.name
}

object ModuleDefinition {
  def apply[T <: BaseModule](
    proto:   T,
    circuit: Option[BaseModule]
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): ModuleDefinition[T] = {
    val newChild = Module.do_pseudo_apply(new ModuleDefinition(proto))
    newChild._circuit = circuit
    newChild._parent = None
    newChild
  }
}
