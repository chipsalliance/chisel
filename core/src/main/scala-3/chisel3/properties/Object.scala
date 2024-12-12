// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.experimental.SourceInfo
import chisel3.internal.firrtl.ir.{DefClass, DefObject, Node}
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

object DynamicObject extends ObjectDynamicObjectImpl
