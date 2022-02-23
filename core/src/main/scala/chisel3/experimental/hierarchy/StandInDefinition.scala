// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.language.experimental.macros
import chisel3._

import chisel3.internal.{PseudoModule, HasId}
import chisel3.internal.firrtl._
import chisel3.experimental.hierarchy.core._
import scala.collection.mutable.HashMap
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import firrtl.annotations.{IsModule, ModuleTarget}

/** Represents a Definition root module, when accessing something from a definition
  *
  * @note This is necessary to distinguish between the toTarget behavior for a Module returned from a Definition,
  * versus a normal Module. A normal Module.toTarget will always return a local target. If calling toTarget
  * on a Module returned from a Definition (and thus wrapped in an Instance), we need to return the non-local
  * target whose root is the Definition. This DefinitionClone is used to represent the root parent of the
  * InstanceClone (which represents the returned module).
  */
private[chisel3] final case class StandInDefinition[T <: BaseModule](proto: T, circuit: Option[BaseModule]) extends PseudoModule with IsStandIn[T] {
  override def equals(a: Any): Boolean = {
    a match {
      case d: StandInDefinition[_] if d.proto == proto && d.circuit == circuit => true
      case _ => false
    }
  }
  val parent = None
  _parent = None
  _circuit = circuit

  def toInstance:   core.Instance[T] = new core.Instance(StandIn(this))
  def toDefinition: core.Definition[T] = new core.Definition(StandIn(this))

  // ======== THINGS TO MAKE CHISEL WORK ========

  //override def toString = s"StandInDefinition($proto, $circuit)"
  // No addition components are generated
  private[chisel3] def generateComponent(): Option[Component] = None
  // Do not call default addId function, which may modify a module that is already "closed"
  override def addId(d: HasId): Unit = ()
  // Necessary for toTarget to work
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
  // Module name is the same as proto's module name
  override def desiredName: String = proto.name
}

object StandInDefinition {
  def apply[T <: BaseModule](proto: T, circuit: Option[BaseModule]): StandInDefinition[T] = {
    val newChild = Module.do_pseudo_apply(new StandInDefinition(proto, circuit))(
      chisel3.internal.sourceinfo.UnlocatableSourceInfo,
      chisel3.ExplicitCompileOptions.Strict
    )
    newChild
  }
}