package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.PrimOps.Not
import firrtl.Utils.isTemp
import firrtl.WrappedExpression._

import scala.collection.mutable

object InlineNotsTransform {

  /** Returns true if Expression is a Not PrimOp, false otherwise */
  private def isNot(expr: Expression): Boolean = expr match {
    case DoPrim(Not, args,_,_) => args.forall(isSimpleExpr)
    case _ => false
  }

  // Checks if an Expression is made up of only Nots terminated by a Literal or Reference.
  // private because it's not clear if this definition of "Simple Expression" would be useful elsewhere.
  // Note that this can have false negatives but MUST NOT have false positives.
  private def isSimpleExpr(expr: Expression): Boolean = expr match {
    case _: WRef | _: Literal | _: WSubField => true
    case DoPrim(Not, args, _,_) => args.forall(isSimpleExpr)
    case _ => false
  }

  /** Mapping from references to the [[firrtl.ir.Expression Expression]]s that drive them */
  type Netlist = mutable.HashMap[WrappedExpression, Expression]

  /** Recursively replace [[WRef]]s with new [[Expression]]s
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[firrtl.ir.DefNode DefNode]]s to their connected
    * [[firrtl.ir.Expression Expression]]s. It is '''not''' mutated in this function
    * @param expr the Expression being transformed
    * @return Returns expr with Nots inlined
    */
  def onExpr(netlist: Netlist)(expr: Expression): Expression = {
    expr.map(onExpr(netlist)) match {
      case e @ WRef(name, _,_,_) =>
        netlist.get(we(e))
               .filter(isNot)
               .getOrElse(e)
      // replace back-to-back inversions with a straight rename
      case lhs @ DoPrim(Not, Seq(inv), _,_) if isSimpleExpr(inv) =>
        netlist.getOrElse(we(inv), inv) match {
          case DoPrim(Not, Seq(rhs), _,_) if isSimpleExpr(inv) => rhs
          case _ => lhs  // Not a candiate
        }
      case other => other // Not a candidate
    }
  }

  /** Inline nots in a Statement
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[firrtl.ir.DefNode DefNode]]s to their connected
    * [[firrtl.ir.Expression Expression]]s. This function '''will''' mutate it if stmt is a [[firrtl.ir.DefNode
    * DefNode]] with a value that is a [[PrimOp]] Not
    * @param stmt the Statement being searched for nodes and transformed
    * @return Returns stmt with nots inlined
    */
  def onStmt(netlist: Netlist)(stmt: Statement): Statement =
    stmt.map(onStmt(netlist)).map(onExpr(netlist)) match {
      case node @ DefNode(_, name, value) if isTemp(name) =>
        netlist(we(WRef(name))) = value
        node
      case other => other
    }

  /** Inline nots in a Module */
  def onMod(mod: DefModule): DefModule = mod.map(onStmt(new Netlist))
}

/** Inline nodes that are simple nots */
class InlineNotsTransform extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(InlineNotsTransform.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
