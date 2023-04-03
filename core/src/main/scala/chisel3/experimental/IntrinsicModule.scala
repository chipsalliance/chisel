// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3.SpecifiedDirection
import chisel3.experimental.{BaseModule, Param}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import scala.annotation.nowarn

private[chisel3] abstract class BaseIntrinsicModule(intrinsicName: String) extends BaseModule {
  val intrinsic = intrinsicName
}

@nowarn("msg=class Port") // delete when Port becomes private
abstract class IntrinsicModule(intrinsicName: String, val params: Map[String, Param] = Map.empty[String, Param])
    extends BaseIntrinsicModule(intrinsicName) {
  private[chisel3] override def generateComponent(): Option[Component] = {
    require(!_closed, "Can't generate intmodule more than once")
    _closed = true

    // Ports are named in the same way as regular Modules
    namePorts()

    val firrtlPorts = getModulePorts.map {
      case port => Port(port, port.specifiedDirection, UnlocatableSourceInfo)
    }
    val component = DefIntrinsicModule(this, name, firrtlPorts, SpecifiedDirection.Unspecified, params)
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
  }
}
