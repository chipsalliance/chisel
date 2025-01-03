// SPDX-License-Identifier: Apache-2.0

package chisel3
package simulator

import svsim._

case class DutContext(clock: Option[Clock], ports: Seq[(Data, ModuleInfo.Port)], maxWaitCycles: Int = 1000)

object DutContext {
  private val dynamicVariable = new scala.util.DynamicVariable[Option[DutContext]](None)
  def withValue[T](dut: DutContext)(body: => T): T = {
    require(dynamicVariable.value.isEmpty, "Nested test contexts are not supported.")
    dynamicVariable.withValue(Some(dut))(body)
  }
  def current: DutContext = dynamicVariable.value.get
}
