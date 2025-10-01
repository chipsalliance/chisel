// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3.SpecifiedDirection
import chisel3.experimental.{BaseModule, Param}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._
import chisel3.layer.Layer

private[chisel3] abstract class BaseIntrinsicModule(intrinsicName: String) extends BaseModule {
  val intrinsic = intrinsicName
}

@deprecated("use Intrinsic and IntrinsicExpr instead, intrinsic modules are deprecated", "Chisel 7.0.0")
abstract class IntrinsicModule(intrinsicName: String, val params: Map[String, Param] = Map.empty[String, Param])
    extends BaseIntrinsicModule(intrinsicName) {
  private[chisel3] override def generateComponent(): Option[Component] = {
    require(!_closed, "Can't generate intmodule more than once")
    evaluateAtModuleBodyEnd()
    _closed = true

    // Ports are named in the same way as regular Modules
    namePorts()

    val firrtlPorts = getModulePortsAndLocators.map { case (port, _, associations) =>
      Port(port, port.specifiedDirection, associations, UnlocatableSourceInfo)
    }
    val component = DefIntrinsicModule(this, name, firrtlPorts, SpecifiedDirection.Unspecified, params)
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo
  }
}
