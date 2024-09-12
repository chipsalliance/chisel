// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.{Module, RawModule, SpecifiedDirection}
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.firrtl.ir.{DefClass, DefObject, Node}
import chisel3.internal.{throwException, Builder, HasId, NamedComponent}
import chisel3.internal.binding.ObjectFieldBinding

import scala.collection.immutable.HashMap
import scala.language.existentials

private[chisel3] trait DynamicObjectImpl extends HasId with NamedComponent {

  val className: ClassType

  private val tpe = Property[className.Type]()

  _parent.foreach(_.addId(this))

  // Keep state for a reference to the Class from which the DynamicObject was created.
  // This is used to update the Class ref to the DynamicObject ref, for Classes created via DynamicObject.apply.
  private var _class: Class = null
  protected[chisel3] def setSourceClass(cls: Class): Unit = {
    require(_class == null, "Cannot set DynamicObject class multiple times")
    _class = cls
  }
  private def getSourceClass: Option[Class] = Option(_class)

  /** Set the source Class ref to this DynamicObject's ref.
    *
    * After the DynamicObject is named, this must be called so the Class ref will be updated to the DynamicObject ref.
    * This is needed for any secret ports that are bored in the class, which point to the Class ref.
    */
  protected[chisel3] def setSourceClassRef(): Unit = {
    getSourceClass.foreach(_.setRef(this.getRef, true))
  }

  /** Get a reference to this Object, suitable for use Ports.
    */
  def getReference: Property[ClassType] = tpe

  /** Get a field from this Object.
    *
    * *WARNING*: It is the caller's responsibility to ensure the field exists, with the correct type and direction.
    */
  def getField[T](name: String)(implicit tpe: PropertyType[T]): Property[tpe.Type] = {
    val field = Property[T]()
    field.setRef(Node(this), name)
    field.bind(ObjectFieldBinding(_parent.get), SpecifiedDirection.Unspecified)
    field
  }

  def getField[T](name: String, property: Property[T]): Property[T] = {
    val field = property.cloneType
    field.setRef(Node(this), name)
    field.bind(ObjectFieldBinding(_parent.get), SpecifiedDirection.Unspecified)
    field
  }
}

private[chisel3] trait ObjectDynamicObjectImpl {

  protected def _applyImpl[T <: Class](bc: => T)(implicit sourceInfo: SourceInfo): DynamicObject = {
    // Instantiate the Class definition.
    val cls = Module.evaluate[T](bc)

    // Build the DynamicObject with associated bindings.
    val obj = Class.unsafeGetDynamicObject(cls.name)

    // Save the source Class for this DynamicObject.
    // After the DynamicObject is named, the Class ref will be updated to the DynamicObject ref.
    // This is needed for any secret ports that are bored in the class, which point to the Class ref.
    obj.setSourceClass(cls)

    obj
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
