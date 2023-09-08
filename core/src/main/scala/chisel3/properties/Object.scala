// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.SpecifiedDirection
import chisel3.internal.{throwException, HasId, NamedComponent, ObjectFieldBinding}

import scala.collection.immutable.HashMap
import scala.language.existentials

/** Represents an instance of a Class.
  *
  * This cannot be instantiated directly, instead see Class.unsafeGetDynamicObject.
  *
  * The DynamicObject is generally unsafe, in that its getField method does not check the name, type, or direction of
  * the accessed field. It may be used with care, and a more typesafe version will be added.
  */
class DynamicObject private[chisel3] (val className: ClassType) extends HasId with NamedComponent {
  private val tpe = Property[className.Type]()

  _parent.foreach(_.addId(this))

  /** Get a reference to this Object, suitable for use Ports.
    */
  def getReference: Property[ClassType] = tpe

  /** Get a field from this Object.
    *
    * *WARNING*: It is the caller's responsibility to ensure the field exists, with the correct type and direction.
    */
  def getField[T](name: String)(implicit tpe: PropertyType[T]): Property[tpe.Type] = {
    val field = Property[T]()
    field.setRef(this, name)
    field.bind(ObjectFieldBinding(_parent.get), SpecifiedDirection.Unspecified)
    field
  }

  def getField[T](name: String, property: Property[T]): Property[T] = {
    val field = property.cloneType
    field.setRef(this, name)
    field.bind(ObjectFieldBinding(_parent.get), SpecifiedDirection.Unspecified)
    field
  }
}
