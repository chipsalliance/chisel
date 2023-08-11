// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.SpecifiedDirection
import chisel3.internal.{throwException, HasId, NamedComponent, ObjectFieldBinding}

import scala.collection.immutable.HashMap
import scala.language.dynamics

class Object(cls: Class) extends HasId with NamedComponent with Dynamic {
  private val tpe = Property(cls)

  _parent.foreach(_.addId(this))

  def getReference: Property[ClassType] = tpe

  def selectDynamic[T: PropertyType](name: String): Property[T] = {
    val fieldOpt = cls.getField[T](name)

    if (!fieldOpt.isDefined) {
      throwException(s"Field $name does not exist on Class ${cls.name}")
    }

    val field = fieldOpt.get.cloneTypeFull
    field.setRef(this, name)
    field.bind(ObjectFieldBinding(_parent.get), SpecifiedDirection.Unspecified)
    field
  }
}
