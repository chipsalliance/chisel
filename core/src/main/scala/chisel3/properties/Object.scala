// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.SpecifiedDirection
import chisel3.experimental.BaseModule
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

/** Represents an instance of a Class.
  *
  * This exists to associate an Instance[Class] with a Property[ClassType] for that Class.
  *
  * After the instance's ModuleClone has been named, the StaticObject and underlying Property[ClassType] have their ref
  * set to the ModuleClone's ref.
  */
private[chisel3] class StaticObject(baseModule: BaseModule) extends HasId with NamedComponent {
  private val tpe = Class.unsafeGetReferenceType(baseModule.name)

  _parent.foreach(_.addId(this))

  /** Get a reference to this Object, suitable for use in Ports or supported Property collections.
    */
  def getReference: Property[ClassType] = tpe

  /** Get the underlying BaseModule of the Instance[Class] for this Object.
    */
  def getInstanceModule: BaseModule = baseModule
}
