// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import scala.language.experimental.macros

import chisel3.{Module, RawModule, SpecifiedDirection}
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.firrtl.ir.{DefClass, DefObject}
import chisel3.internal.sourceinfo.InstTransform
import chisel3.internal.{throwException, Builder, HasId, NamedComponent, ObjectFieldBinding}

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

  // Keep state for a reference to the Class from which the DynamicObject was created.
  // This is used to update the Class ref to the DynamicObject ref, for Classes created via DynamicObject.apply.
  private var _class: Option[Class] = None
  protected[chisel3] def setSourceClass(cls: Class): Unit = {
    require(!_class.isDefined, "Cannot set DynamicObject class multiple times")
    _class = Some(cls)
  }
  protected[chisel3] def getSourceClass: Option[Class] = _class

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

object DynamicObject {

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
  def do_apply[T <: Class](bc: => T)(implicit sourceInfo: SourceInfo): DynamicObject = {
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
