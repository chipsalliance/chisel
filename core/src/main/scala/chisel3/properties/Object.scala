// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.SpecifiedDirection
import chisel3.internal.{throwException, HasId, NamedComponent, ObjectFieldBinding}

import scala.collection.immutable.HashMap

case class DynamicObject private[chisel3] (className: String) extends HasId with NamedComponent {
  private val tpe = new Property[ClassType](Some(ClassType(className)))

  _parent.foreach(_.addId(this))

  def getReference: Property[ClassType] = tpe
}
