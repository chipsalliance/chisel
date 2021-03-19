// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.PrimOps.Pad
import firrtl.options.Dependency

import firrtl.Utils.{isBitExtract, isCast, NodeMap}

object InlineCastsTransform {

  // Checks if an Expression is made up of only casts terminated by a Literal or Reference
  // There must be at least one cast
  // Note that this can have false negatives but MUST NOT have false positives
  private def isSimpleCast(castSeen: Boolean)(expr: Expression): Boolean = expr match {
    case _: WRef | _: Literal | _: WSubField => castSeen
    case DoPrim(op, args, _, _) if isCast(op) => args.forall(isSimpleCast(true))
    case _                                    => false
  }

  /** Recursively replace [[WRef]]s with new [[firrtl.ir.Expression Expression]]s
    *
    * @param replace a '''mutable''' HashMap mapping [[WRef]]s to values with which the [[WRef]]
    * will be replaced. It is '''not''' mutated in this function
    * @param expr the Expression being transformed
    * @return Returns expr with [[WRef]]s replaced by values found in replace
    */
  def onExpr(replace: NodeMap)(expr: Expression): Expression = {
    // Keep track if we've seen any non-cast expressions while recursing
    def rec(hasNonCastParent: Boolean)(expr: Expression): Expression = expr match {
      // Skip pads to avoid inlining literals into pads which results in invalid Verilog
      case DoPrim(op, _, _, _) if (isBitExtract(op) || op == Pad) => expr
      case e =>
        e.map(rec(hasNonCastParent || !isCast(e))) match {
          case e @ WRef(name, _, _, _) =>
            replace
              .get(name)
              .filter(isSimpleCast(castSeen = false))
              .getOrElse(e)
          case e @ DoPrim(op, Seq(WRef(name, _, _, _)), _, _) if isCast(op) =>
            replace
              .get(name)
              // Only inline the Expression if there is no non-cast parent in the expression tree OR
              // if the subtree contains only casts and references.
              .filter(x => !hasNonCastParent || isSimpleCast(castSeen = true)(x))
              .map(value => e.copy(args = Seq(value)))
              .getOrElse(e)
          case other => other // Not a candidate
        }
    }
    rec(false)(expr)
  }

  /** Inline casts in a Statement
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[firrtl.ir.DefNode DefNode]]s to their connected
    * [[firrtl.ir.Expression Expression]]s. This function '''will''' mutate
    * it if stmt is a [[firrtl.ir.DefNode DefNode]]
    * with a value that is a cast [[firrtl.ir.PrimOp PrimpOp]]
    * @param stmt the Statement being searched for nodes and transformed
    * @return Returns stmt with casts inlined
    */
  def onStmt(netlist: NodeMap)(stmt: Statement): Statement =
    stmt.map(onStmt(netlist)).map(onExpr(netlist)) match {
      case node @ DefNode(_, name, value) =>
        netlist(name) = value
        node
      case other => other
    }

  /** Replaces truncating arithmetic in a Module */
  def onMod(mod: DefModule): DefModule = mod.map(onStmt(new NodeMap))
}

/** Inline nodes that are simple casts */
class InlineCastsTransform extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq(
      Dependency[BlackBoxSourceHelper],
      Dependency[FixAddingNegativeLiterals],
      Dependency[ReplaceTruncatingArithmetic],
      Dependency[InlineBitExtractionsTransform],
      Dependency[PropagatePresetAnnotations]
    )

  override def optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform): Boolean = a match {
    case _: LegalizeClocksTransform => true
    case _ => false
  }

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(InlineCastsTransform.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
