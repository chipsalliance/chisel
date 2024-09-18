// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import scala.language.experimental.macros

import chisel3.experimental.SourceInfo
import chisel3.internal.firrtl.ir.{DefClass, DefObject, Node}
import chisel3.internal.sourceinfo.InstTransform
import chisel3.internal.{throwException, Builder, HasId, NamedComponent}
import chisel3.internal.binding.ObjectFieldBinding

/** Represents an instance of a Class.
  *
  * This cannot be instantiated directly, instead see Class.unsafeGetDynamicObject.
  *
  * The DynamicObject is generally unsafe, in that its getField method does not check the name, type, or direction of
  * the accessed field. It may be used with care, and a more typesafe version called StaticObject has been added, which
  * works with the Definition / Instance APIs.
  *
  * To create a DynamicObject directly, wrap a Class with DynamicObject.apply. For example:
  *
  *  {{{
  *    val obj = DynamicObject(new Class {
  *      override def desiredName = "Test"
  *      val in = IO(Input(Property[Int]()))
  *      val out = IO(Output(Property[Int]()))
  *      out := in
  *    })
  *  }}}
  */
class DynamicObject private[chisel3] (val className: ClassType) extends DynamicObjectImpl

object DynamicObject extends ObjectDynamicObjectImpl {

  /** A wrapper method to wrap Class instantiations and return a DynamicObject.
    *
    * This is necessary to help Chisel track internal state. This can be used instead of `Definition.apply` if a
    * DynamicObject is required. If possible, it is safer to user `Definition.apply` and StaticObject.
    *
    * @param bc the Class being created
    *
    * @return a DynamicObject representing an instance of the Class
    */
  def apply[T <: Class](bc: => T): DynamicObject = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Class](bc: => T)(implicit sourceInfo: SourceInfo): DynamicObject = _applyImpl(bc)
}
