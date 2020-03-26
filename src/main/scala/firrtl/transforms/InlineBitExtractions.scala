// See LICENSE for license details.

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{Dependency, PreservesAll}
import firrtl.PrimOps.{Bits, Head, Tail, Shr}
import firrtl.Utils.{isBitExtract, isTemp}
import firrtl.WrappedExpression._

import scala.collection.mutable

object InlineBitExtractionsTransform {

  // Checks if an Expression is made up of only Bits terminated by a Literal or Reference.
  // private because it's not clear if this definition of "Simple Expression" would be useful elsewhere.
  // Note that this can have false negatives but MUST NOT have false positives.
  private def isSimpleExpr(expr: Expression): Boolean = expr match {
    case _: WRef | _: Literal | _: WSubField => true
    case DoPrim(op, args, _,_) if isBitExtract(op) => args.forall(isSimpleExpr)
    case _ => false
  }

  // replace Head/Tail/Shr with Bits for easier back-to-back Bits Extractions
  private def lowerToDoPrimOpBits(expr: Expression): Expression = expr match {
    case DoPrim(Head, rhs, c, tpe) if isSimpleExpr(expr) =>
      val msb = bitWidth(rhs.head.tpe) - 1
      val lsb = bitWidth(rhs.head.tpe) - c.head
      DoPrim(Bits, rhs, Seq(msb,lsb), tpe)
    case DoPrim(Tail, rhs, c, tpe) if isSimpleExpr(expr) =>
      val msb = bitWidth(rhs.head.tpe) - c.head - 1
      DoPrim(Bits, rhs, Seq(msb,0), tpe)
    case DoPrim(Shr, rhs, c, tpe) if isSimpleExpr(expr) =>
      DoPrim(Bits, rhs, Seq(bitWidth(rhs.head.tpe)-1, c.head), tpe)
    case _ => expr // Not a candidate
  }

  /** Mapping from references to the [[firrtl.ir.Expression Expression]]s that drive them */
  type Netlist = mutable.HashMap[WrappedExpression, Expression]

  /** Recursively replace [[WRef]]s with new [[firrtl.ir.Expression Expression]]s
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[firrtl.ir.DefNode DefNode]]s to their connected
    * [[firrtl.ir.Expression Expression Expression]]s. It is '''not''' mutated in this function
    * @param expr the Expression being transformed
    * @return Returns expr with Bits inlined
    */
  def onExpr(netlist: Netlist)(expr: Expression): Expression = {
    expr.map(onExpr(netlist)) match {
      case e @ WRef(name, _,_,_) =>
        netlist.get(we(e))
               .filter(isBitExtract)
               .getOrElse(e)
      // replace back-to-back Bits Extractions
      case lhs @ DoPrim(lop, ival, lc, ltpe) if isSimpleExpr(lhs) =>
        ival.head match {
          case of @ DoPrim(rop, rhs, rc, rtpe) if isSimpleExpr(of) =>
            (lop, rop) match {
              case (Head, Head) => DoPrim(Head, rhs, Seq(lc.head min rc.head), ltpe)
              case (Tail, Tail) => DoPrim(Tail, rhs, Seq(lc.head + rc.head), ltpe)
              case (Shr,  Shr)  => DoPrim(Shr,  rhs, Seq(lc.head + rc.head), ltpe)
              case (_,_) => (lowerToDoPrimOpBits(lhs), lowerToDoPrimOpBits(of)) match {
                case (DoPrim(Bits, _, Seq(lmsb, llsb), _), DoPrim(Bits, _, Seq(rmsb, rlsb), _)) =>
                  DoPrim(Bits, rhs, Seq(lmsb+rlsb,llsb+rlsb), ltpe)
                case (_,_) => lhs  // Not a candidate
              }
            }
            case _ => lhs  // Not a candidate
          }
      case other => other // Not a candidate
    }
  }

  /** Inline bits in a Statement
    *
    * @param netlist a '''mutable''' HashMap mapping references to [[firrtl.ir.DefNode DefNode]]s to their connected
    * [[firrtl.ir.Expression Expression]]s. This function '''will''' mutate it if stmt is
    * a [[firrtl.ir.DefNode DefNode]] with a Temporary name and a value that is a [[firrtl.ir.PrimOp PrimOp]] Bits
    * @param stmt the Statement being searched for nodes and transformed
    * @return Returns stmt with Bits inlined
    */
  def onStmt(netlist: Netlist)(stmt: Statement): Statement =
    stmt.map(onStmt(netlist)).map(onExpr(netlist)) match {
      case node @ DefNode(_, name, value) if isTemp(name) =>
        netlist(we(WRef(name))) = value
        node
      case other => other
    }

  /** Replaces bits in a Module */
  def onMod(mod: DefModule): DefModule = mod.map(onStmt(new Netlist))
}

/** Inline nodes that are simple bits */
class InlineBitExtractionsTransform extends Transform with PreservesAll[Transform] {
  def inputForm = UnknownForm
  def outputForm = UnknownForm

  override val prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq( Dependency[BlackBoxSourceHelper],
         Dependency[FixAddingNegativeLiterals],
         Dependency[ReplaceTruncatingArithmetic] )

  override val optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override val dependents = Seq.empty

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(InlineBitExtractionsTransform.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
