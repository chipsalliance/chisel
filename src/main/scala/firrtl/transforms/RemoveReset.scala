// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.traversals.Foreachers._
import firrtl.WrappedExpression.we
import firrtl.options.Dependency

import scala.collection.{immutable, mutable}

/** Remove Synchronous Reset
  *
  * @note This pass must run after LowerTypes
  */
object RemoveReset extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.MidForm ++
    Seq( Dependency(passes.LowerTypes),
         Dependency(passes.Legalize) )

  override def optionalPrerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform): Boolean = a match {
    case firrtl.passes.ResolveFlows => true
    case _                          => false
  }

  private case class Reset(cond: Expression, value: Expression, info: Info)

  /** Return an immutable set of all invalid expressions in a module
    * @param m a module
    */
  private def computeInvalids(m: DefModule): immutable.Set[WrappedExpression] = {
    val invalids = mutable.HashSet.empty[WrappedExpression]

    def onStmt(s: Statement): Unit = s match {
      case IsInvalid(_, expr)                                 => invalids += we(expr)
      case Connect(_, lhs, rhs) if invalids.contains(we(rhs)) => invalids += we(lhs)
      case other                                              => other.foreach(onStmt)
    }

    m.foreach(onStmt)
    invalids.toSet
  }

  private def onModule(m: DefModule): DefModule = {
    val resets = mutable.HashMap.empty[String, Reset]
    val invalids = computeInvalids(m)
    def onStmt(stmt: Statement): Statement = {
      stmt match {
        /* A register is initialized to an invalid expression */
        case reg @ DefRegister(_, _, _, _, _, init) if invalids.contains(we(init)) =>
          reg.copy(reset = Utils.zero, init = WRef(reg))
        case reg @ DefRegister(_, rname, _, _, Utils.zero, _) =>
          reg.copy(init = WRef(reg)) // canonicalize
        case reg @ DefRegister(info , rname, _, _, reset, init) if reset.tpe != AsyncResetType =>
          // Add register reset to map
          resets(rname) = Reset(reset, init, info)
          reg.copy(reset = Utils.zero, init = WRef(reg))
        case Connect(info, ref @ WRef(rname, _, RegKind, _), expr) if resets.contains(rname) =>
          val reset = resets(rname)
          val muxType = Utils.mux_type_and_widths(reset.value, expr)
          // Use reg source locator for mux enable and true value since that's where they're defined
          val infox = MultiInfo(reset.info, reset.info, info)
          Connect(infox, ref, Mux(reset.cond, reset.value, expr, muxType))
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
