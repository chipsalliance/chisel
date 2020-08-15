// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.Utils._
import firrtl.options.Dependency
import firrtl.InfoExpr.{orElse, unwrap}

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
    // We want to flatten Mux trees for reg updates into if-trees for improved QoR for conditional
    // updates.  Sometimes the fan-in for a register has a mux structure with repeated
    // sub-expressions that are themselves complex mux structures. These repeated structures can
    // cause explosions in the size and complexity of the Verilog. In addition, user code that
    // follows such structure often will have conditions in the sub-trees that are mutually
    // exclusive with the conditions in the muxes closer to the register input. For example:
    //
    // when a :      ; when 1
    //   r <= foo
    // when b :      ; when 2
    //   when a :
    //     r <= bar  ; when 3
    //
    // After expand whens, when 1 is a common sub-expression that will show up twice in the mux
    // structure from when 2:
    //
    // _GEN_0 = mux(a, foo, r)
    // _GEN_1 = mux(a, bar, _GEN_0)
    // r <= mux(b, _GEN_1, _GEN_0)
    //
    // Inlining _GEN_0 into _GEN_1 would result in unreachable lines in the Verilog. While we could
    // do some optimizations here, this is *not* really a problem, it's just that Verilog metrics
    // are based on the assumption of human-written code and as such it results in unreachable
    // lines. Simply not inlining avoids this issue and leaves the optimizations up to synthesis
    // tools which do a great job here.
    val maxDepth = 4

    val regUpdates = mutable.ArrayBuffer.empty[Connect]
    val netlist = buildNetlist(mod)

    // First traversal marks expression that would be inlined multiple times as endpoints
    // Note that we could traverse more than maxDepth times - this corresponds to an expression that
    // is already a very deeply nested mux
    def determineEndpoints(expr: Expression): collection.Set[WrappedExpression] = {
      val seen = mutable.HashSet.empty[WrappedExpression]
      val endpoint = mutable.HashSet.empty[WrappedExpression]
      def rec(depth: Int)(e: Expression): Unit = {
        val (_, ex) = kind(e) match {
          case NodeKind | WireKind if depth < maxDepth && !seen(e) =>
            seen += e
            unwrap(netlist.getOrElse(e, e))
          case _ => unwrap(e)
        }
        ex match {
          case Mux(_, tval, fval, _) =>
            rec(depth + 1)(tval)
            rec(depth + 1)(fval)
          case _ =>
            // Mark e not ex because original reference is the endpoint, not op or whatever
            endpoint += ex
        }
      }
      rec(0)(expr)
      endpoint
    }

    def constructRegUpdate(start: Expression): (Info, Expression) = {
      val endpoints = determineEndpoints(start)
      def rec(e: Expression): (Info, Expression) = {
        val (info, expr) = kind(e) match {
          case NodeKind | WireKind if !endpoints(e) => unwrap(netlist.getOrElse(e, e))
          case _                                    => unwrap(e)
        }
        expr match {
          case Mux(cond, tval, fval, tpe) =>
            val (tinfo, tvalx) = rec(tval)
            val (finfo, fvalx) = rec(fval)
            val infox = combineInfos(info, tinfo, finfo)
            (infox, Mux(cond, tvalx, fvalx, tpe))
          // Return the original expression to end flattening
          case _ => unwrap(e)
        }
      }
      rec(start)
    }

    def onStmt(stmt: Statement): Statement = stmt.map(onStmt) match {
      case reg @ DefRegister(_, rname, _, _, resetCond, _) =>
        assert(
          resetCond.tpe == AsyncResetType || resetCond == Utils.zero,
          "Synchronous reset should have already been made explicit!"
        )
        val ref = WRef(reg)
        val (info, rhs) = constructRegUpdate(netlist.getOrElse(ref, ref))
        val update = Connect(info, ref, rhs)
        regUpdates += update
        reg
      // Remove connections to Registers so we preserve LowFirrtl single-connection semantics
      case Connect(_, lhs, _) if kind(lhs) == RegKind => EmptyStmt
      case other                                      => other
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
    Seq(
      Dependency[BlackBoxSourceHelper],
      Dependency[FixAddingNegativeLiterals],
      Dependency[ReplaceTruncatingArithmetic],
      Dependency[InlineBitExtractionsTransform],
      Dependency[InlineCastsTransform],
      Dependency[LegalizeClocksTransform]
    )

  override def optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform): Boolean = a match {
    case _: DeadCodeElimination => true
    case _ => false
  }

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map {
      case mod: Module    => FlattenRegUpdate.flattenReg(mod)
      case ext: ExtModule => ext
    }
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
