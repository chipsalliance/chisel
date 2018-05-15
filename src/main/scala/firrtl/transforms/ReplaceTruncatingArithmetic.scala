package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._

import scala.collection.mutable

object ReplaceTruncatingArithmetic {

  /** Mapping from references to the [[Expression]]s that drive them */
  type Netlist = mutable.HashMap[WrappedExpression, Expression]

  private val SeqBIOne = Seq(BigInt(1))

  /** Replaces truncating arithmetic in an Expression
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[DefNode]]s to their connected
    * [[Expression]]s. It is '''not''' mutated in this function
    * @param expr the Expression being transformed
    * @return Returns expr with truncating arithmetic replaced
    */
  def onExpr(netlist: Netlist)(expr: Expression): Expression =
    expr.map(onExpr(netlist)) match {
      case orig @ DoPrim(Tail, Seq(e), SeqBIOne, tailtpe) =>
        netlist.getOrElse(we(e), e) match {
          case DoPrim(Add, args, cs, _) => DoPrim(Addw, args, cs, tailtpe)
          case DoPrim(Sub, args, cs, _) => DoPrim(Subw, args, cs, tailtpe)
          case _ => orig // Not a candidate
        }
      case other => other // Not a candidate
    }

  /** Replaces truncating arithmetic in a Statement
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[DefNode]]s to their connected
    * [[Expression]]s. This function '''will''' mutate it if stmt contains a [[DefNode]]
    * @param stmt the Statement being searched for nodes and transformed
    * @return Returns stmt with truncating arithmetic replaced
    */
  def onStmt(netlist: Netlist)(stmt: Statement): Statement =
    stmt.map(onStmt(netlist)).map(onExpr(netlist)) match {
      case node @ DefNode(_, name, value) =>
        netlist(we(WRef(name))) = value
        node
      case other => other
    }

  /** Replaces truncating arithmetic in a Module */
  def onMod(mod: DefModule): DefModule = mod.map(onStmt(new Netlist))
}

/** Replaces non-expanding arithmetic
  *
  * In the case where the result of `add` or `sub` immediately throws away the expanded msb, this
  * transform will replace the operation with a non-expanding operator `addw` or `subw`
  * respectively.
  *
  * @note This replaces some FIRRTL primops with ops that are not actually legal FIRRTL. They are
  * useful for emission to languages that support non-expanding arithmetic (like Verilog)
  */
class ReplaceTruncatingArithmetic extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(ReplaceTruncatingArithmetic.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}

