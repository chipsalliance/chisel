// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.Utils._
import firrtl.options.Dependency
import firrtl.InfoExpr.orElse

import scala.collection.mutable

object FlattenRegUpdate {

  // Combination function for dealing with inlining of muxes and the handling of Triples of infos
  private def combineInfos(muxInfo: Info, tinfo: Info, finfo: Info): Info = {
    val (eninfo, tinfoAlt, finfoAlt) = MultiInfo.demux(muxInfo)
    // Use MultiInfo constructor to preserve NoInfos
    new MultiInfo(List(eninfo, orElse(tinfo, tinfoAlt), orElse(finfo, finfoAlt)))
  }

  /** Mapping from references to the [[firrtl.ir.Expression Expression]]s that drive them */
  type Netlist = mutable.HashMap[WrappedExpression, Expression]

  /** Build a [[Netlist]] from a Module's connections and Nodes
    *
    * This assumes [[firrtl.LowForm LowForm]]
    *
    * @param mod [[firrtl.ir.Module Module]] from which to build a [[Netlist]]
    * @return [[Netlist]] of the module's connections and nodes
    */
  def buildNetlist(mod: Module): Netlist = {
    val netlist = new Netlist()
    def onStmt(stmt: Statement): Statement = {
      stmt.map(onStmt) match {
        case Connect(info, lhs, rhs) =>
          val expr = if (info == NoInfo) rhs else InfoExpr(info, rhs)
          netlist(lhs) = expr
        case DefNode(info, nname, rhs) =>
          val expr = if (info == NoInfo) rhs else InfoExpr(info, rhs)
          netlist(WRef(nname)) = expr
        case _: IsInvalid => throwInternalError("Unexpected IsInvalid, should have been removed by now")
        case _ => // Do nothing
      }
      stmt
    }
    mod.map(onStmt)
    netlist
  }

  /** Flatten Register Updates
    *
    * Constructs nested mux trees (up to a certain arbitrary threshold) for register updates. This
    * can result in dead code that this function does NOT remove.
    *
    * @param mod [[firrtl.ir.Module Module]] to transform
    * @return [[firrtl.ir.Module Module]] with register updates flattened
    */
  def flattenReg(mod: Module): Module = {
    // We want to flatten Mux trees for reg updates into if-trees for
    // improved QoR for conditional updates.  However, unbounded recursion
    // would take exponential time, so don't redundantly flatten the same
    // Mux more than a bounded number of times, preserving linear runtime.
    // The threshold is empirical but ample.
    val flattenThreshold = 4
    val numTimesFlattened = mutable.HashMap[Mux, Int]()
    def canFlatten(m: Mux): Boolean = {
      val n = numTimesFlattened.getOrElse(m, 0)
      numTimesFlattened(m) = n + 1
      n < flattenThreshold
    }

    val regUpdates = mutable.ArrayBuffer.empty[Connect]
    val netlist = buildNetlist(mod)

    def constructRegUpdate(e: Expression): (Info, Expression) = {
      import InfoExpr.unwrap
      // Only walk netlist for nodes and wires, NOT registers or other state
      val (info, expr) = kind(e) match {
        case NodeKind | WireKind => unwrap(netlist.getOrElse(e, e))
        case _ => unwrap(e)
      }
      expr match {
        case mux: Mux if canFlatten(mux) =>
          val (tinfo, tvalx) = constructRegUpdate(mux.tval)
          val (finfo, fvalx) = constructRegUpdate(mux.fval)
          val infox = combineInfos(info, tinfo, finfo)
          (infox, mux.copy(tval = tvalx, fval = fvalx))
        // Return the original expression to end flattening
        case _ => unwrap(e)
      }
    }

    def onStmt(stmt: Statement): Statement = stmt.map(onStmt) match {
      case reg @ DefRegister(_, rname, _,_, resetCond, _) =>
        assert(resetCond.tpe == AsyncResetType || resetCond == Utils.zero,
          "Synchronous reset should have already been made explicit!")
        val ref = WRef(reg)
        val (info, rhs) = constructRegUpdate(netlist.getOrElse(ref, ref))
        val update = Connect(info, ref, rhs)
        regUpdates += update
        reg
      // Remove connections to Registers so we preserve LowFirrtl single-connection semantics
      case Connect(_, lhs, _) if kind(lhs) == RegKind => EmptyStmt
      case other => other
    }

    val bodyx = onStmt(mod.body)
    mod.copy(body = Block(bodyx +: regUpdates.toSeq))
  }

}

/** Flatten register update
  *
  * This transform flattens register updates into a single expression on the rhs of connection to
  * the register
  */
// TODO Preserve source locators
class FlattenRegUpdate extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq( Dependency[BlackBoxSourceHelper],
         Dependency[FixAddingNegativeLiterals],
         Dependency[ReplaceTruncatingArithmetic],
         Dependency[InlineBitExtractionsTransform],
         Dependency[InlineCastsTransform],
         Dependency[LegalizeClocksTransform] )

  override def optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform): Boolean = a match {
    case _: DeadCodeElimination => true
    case _ => false
  }

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map {
      case mod: Module => FlattenRegUpdate.flattenReg(mod)
      case ext: ExtModule => ext
    }
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
