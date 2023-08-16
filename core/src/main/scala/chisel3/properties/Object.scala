// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.SpecifiedDirection
import chisel3.internal.{throwException, HasId, NamedComponent, ObjectFieldBinding}

import scala.collection.immutable.HashMap

/** Represents an instance of a Class.
  *
  * This cannot be instantiated directly, instead see Class.unsafeGetDynamicObject.
  *
  * The DynamicObject is generally unsafe, in that its getField method does not check the name, type, or direction of
  * the accessed field. It may be used with care, and a more typesafe version will be added.
  */
case class DynamicObject private[chisel3] (className: String) extends HasId with NamedComponent {
  private val tpe = new Property[ClassType](Some(ClassType(className)))

  _parent.foreach(_.addId(this))

  /** Get a reference to this Object, suitable for use Ports.
    */
  def getReference: Property[ClassType] = tpe

  /** Get a field from this Object.
    *
    * *WARNING*: It is the caller's responsibility to ensure the field exists, with the correct type and direction.
    */
  def getField[T: PropertyType](name: String): Property[T] = {
    val field = new Property[T](None)
    field.setRef(this, name)
    field.bind(ObjectFieldBinding(_parent.get), SpecifiedDirection.Unspecified)
    field
  }
}
