package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{Dependency, PreservesAll}

import firrtl.Utils.{isCast, NodeMap}

object InlineCastsTransform {

  // Checks if an Expression is made up of only casts terminated by a Literal or Reference
  // There must be at least one cast
  // Note that this can have false negatives but MUST NOT have false positives
  private def isSimpleCast(castSeen: Boolean)(expr: Expression): Boolean = expr match {
    case _: WRef | _: Literal | _: WSubField => castSeen
    case DoPrim(op, args, _,_) if isCast(op) => args.forall(isSimpleCast(true))
    case _ => false
  }

  /** Recursively replace [[WRef]]s with new [[Expression]]s
    *
    * @param replace a '''mutable''' HashMap mapping [[WRef]]s to values with which the [[WRef]]
    * will be replaced. It is '''not''' mutated in this function
    * @param expr the Expression being transformed
    * @return Returns expr with [[WRef]]s replaced by values found in replace
    */
  def onExpr(replace: NodeMap)(expr: Expression): Expression = {
    expr.map(onExpr(replace)) match {
      case e @ WRef(name, _,_,_) =>
        replace.get(name)
               .filter(isSimpleCast(castSeen=false))
               .getOrElse(e)
      case e @ DoPrim(op, Seq(WRef(name, _,_,_)), _,_) if isCast(op) =>
        replace.get(name)
               .map(value => e.copy(args = Seq(value)))
               .getOrElse(e)
      case other => other // Not a candidate
    }
  }

  /** Inline casts in a Statement
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[firrtl.ir.DefNode DefNode]]s to their connected
    * [[firrtl.ir.Expression Expression]]s. This function '''will''' mutate it if stmt is a [[firrtl.ir.DefNode
    * DefNode]] with a value that is a cast [[PrimOp]]
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
class InlineCastsTransform extends Transform with PreservesAll[Transform] {
  def inputForm = UnknownForm
  def outputForm = UnknownForm

  override val prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq( Dependency[BlackBoxSourceHelper],
         Dependency[FixAddingNegativeLiterals],
         Dependency[ReplaceTruncatingArithmetic],
         Dependency[InlineBitExtractionsTransform] )

  override val optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override val dependents = Seq.empty

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(InlineCastsTransform.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
