// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._

import scala.collection.mutable

/** Remove Synchronous Reset
  *
  * @note This pass must run after LowerTypes
  */
class RemoveReset extends Transform {
  def inputForm = MidForm
  def outputForm = MidForm

  private case class Reset(cond: Expression, value: Expression)

  private def onModule(m: DefModule): DefModule = {
    val resets = mutable.HashMap.empty[String, Reset]
    def onStmt(stmt: Statement): Statement = {
      stmt match {
        case reg @ DefRegister(_, rname, _, _, reset, init) if reset != Utils.zero =>
          // Add register reset to map
          resets(rname) = Reset(reset, init)
          reg.copy(reset = Utils.zero, init = WRef(reg))
        case Connect(info, ref @ WRef(rname, _, RegKind, _), expr) if resets.contains(rname) =>
          val reset = resets(rname)
          val muxType = Utils.mux_type_and_widths(reset.value, expr)
          Connect(info, ref, Mux(reset.cond, reset.value, expr, muxType))
        case other => other map onStmt
      }
    }
    m.map(onStmt)
  }

  def execute(state: CircuitState): CircuitState = {
    val c = state.circuit.map(onModule)
    state.copy(circuit = c)
  }
}
