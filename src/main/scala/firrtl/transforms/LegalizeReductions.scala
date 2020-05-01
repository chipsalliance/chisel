package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{Dependency, PreservesAll}
import firrtl.Utils.BoolType


object LegalizeAndReductionsTransform {

  private def allOnesOfType(tpe: Type): Literal = tpe match {
    case UIntType(width @ IntWidth(x)) => UIntLiteral((BigInt(1) << x.toInt) - 1, width)
    case SIntType(width) => SIntLiteral(-1, width)

  }

  def onExpr(expr: Expression): Expression = expr.map(onExpr) match {
    case DoPrim(PrimOps.Andr, Seq(arg), _,_) if bitWidth(arg.tpe) > 64 =>
      DoPrim(PrimOps.Eq, Seq(arg, allOnesOfType(arg.tpe)), Seq(), BoolType)
    case other => other
  }

  def onStmt(stmt: Statement): Statement = stmt.map(onStmt).map(onExpr)

  def onMod(mod: DefModule): DefModule = mod.map(onStmt)
}

/** Turns andr for expression > 64-bit into equality check with all ones
  *
  * Workaround a bug in Verilator v4.026 - v4.032 (inclusive).
  * For context, see https://github.com/verilator/verilator/issues/2300
  */
class LegalizeAndReductionsTransform extends Transform with DependencyAPIMigration with PreservesAll[Transform] {

  override def prerequisites =
    firrtl.stage.Forms.WorkingIR ++
    Seq( Dependency(passes.CheckTypes),
         Dependency(passes.CheckWidths))

  override def optionalPrerequisites = Nil

  override def optionalPrerequisiteOf = Nil

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(LegalizeAndReductionsTransform.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
