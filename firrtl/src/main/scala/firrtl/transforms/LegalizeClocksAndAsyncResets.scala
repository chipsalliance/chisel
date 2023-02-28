// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.Dependency
import firrtl.Utils.isCast

// Fixup otherwise legal Verilog that lint tools and other tools don't like
// Currently:
//   - don't emit "always @(posedge <literal>)"
//     Hitting this case is rare, but legal FIRRTL
// TODO This should be unified with all Verilog legalization transforms
object LegalizeClocksAndAsyncResetsTransform {

  // Checks if an Expression is illegal in use in a @(posedge <Expression>) construct
  // Legality is defined here by what standard lint tools accept
  // Currently only looks for literals nested within casts
  private def isLiteralExpression(expr: Expression): Boolean = expr match {
    case _: Literal => true
    case DoPrim(op, args, _, _) if isCast(op) => args.exists(isLiteralExpression)
    case _                                    => false
  }

  // Wraps the above function to check if a Rest is Async to avoid unneeded
  // hoisting of sync reset literals.
  private def isAsyncResetLiteralExpr(expr: Expression): Boolean =
    if (expr.tpe == AsyncResetType) isLiteralExpression(expr) else false

  /** Legalize Clocks and AsyncResets in a Statement
    *
    * Enforces legal Verilog semantics on all Clock and AsyncReset Expressions.
    * Legal is defined as what standard lint tools accept.
    * Currently only Literal Expressions (guarded by casts) are handled.
    *
    * @note namespace is lazy because it should not typically be needed
    */
  def onStmt(namespace: => Namespace)(stmt: Statement): Statement =
    stmt.map(onStmt(namespace)) match {
      // Proper union types would deduplicate this code
      case r: DefRegister if (isLiteralExpression(r.clock) || isAsyncResetLiteralExpr(r.reset)) =>
        val (clockNodeOpt, rxClock) = if (isLiteralExpression(r.clock)) {
          val node = DefNode(r.info, namespace.newTemp, r.clock)
          (Some(node), r.copy(clock = WRef(node)))
        } else {
          (None, r)
        }
        val (resetNodeOpt, rx) = if (isAsyncResetLiteralExpr(r.reset)) {
          val node = DefNode(r.info, namespace.newTemp, r.reset)
          (Some(node), rxClock.copy(reset = WRef(node)))
        } else {
          (None, rxClock)
        }
        Block(clockNodeOpt ++: resetNodeOpt ++: Seq(rx))
      case Connect(info, loc, rhs @ DoPrim(_, _, _, ClockType)) if (Utils.kind(loc) == MemKind) =>
        val node = DefNode(info, namespace.newTemp, rhs)
        Block(node, Connect(info, loc, WRef(node)))
      case p: Print if isLiteralExpression(p.clk) =>
        val node = DefNode(p.info, namespace.newTemp, p.clk)
        val px = p.copy(clk = WRef(node))
        Block(Seq(node, px))
      case s: Stop if isLiteralExpression(s.clk) =>
        val node = DefNode(s.info, namespace.newTemp, s.clk)
        val sx = s.copy(clk = WRef(node))
        Block(Seq(node, sx))
      case s: Verification if isLiteralExpression(s.clk) =>
        val node = DefNode(s.info, namespace.newTemp, s.clk)
        val sx = s.copy(clk = WRef(node))
        Block(Seq(node, sx))
      case other => other
    }

  def onMod(mod: DefModule): DefModule = {
    // It's actually *extremely* important that this Namespace is a lazy val
    // onStmt accepts it lazily so that we don't perform the namespacing traversal unless necessary
    // If we were to inline the declaration, it would create a Namespace for every problem, causing
    // name collisions
    lazy val namespace = Namespace(mod)
    mod.map(onStmt(namespace))
  }
}

/** Ensure Clocks and AsyncResets to be emitted are legal Verilog */
class LegalizeClocksAndAsyncResetsTransform extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq(
      Dependency[BlackBoxSourceHelper],
      Dependency[FixAddingNegativeLiterals],
      Dependency[ReplaceTruncatingArithmetic],
      Dependency[InlineBitExtractionsTransform],
      Dependency[InlineAcrossCastsTransform]
    )

  override def optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(LegalizeClocksAndAsyncResetsTransform.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
