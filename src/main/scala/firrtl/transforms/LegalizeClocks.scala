package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{Dependency, PreservesAll}
import firrtl.Utils.isCast

// Fixup otherwise legal Verilog that lint tools and other tools don't like
// Currently:
//   - don't emit "always @(posedge <literal>)"
//     Hitting this case is rare, but legal FIRRTL
// TODO This should be unified with all Verilog legalization transforms
object LegalizeClocksTransform {

  // Checks if an Expression is illegal in use in a @(posedge <Expression>) construct
  // Legality is defined here by what standard lint tools accept
  // Currently only looks for literals nested within casts
  private def illegalClockExpr(expr: Expression): Boolean = expr match {
    case _: Literal => true
    case DoPrim(op, args, _,_) if isCast(op) => args.exists(illegalClockExpr)
    case _ => false
  }

  /** Legalize Clocks in a Statement
    *
    * Enforces legal Verilog semantics on all Clock Expressions.
    * Legal is defined as what standard lint tools accept.
    * Currently only Literal Expressions (guarded by casts) are handled.
    *
    * @note namespace is lazy because it should not typically be needed
    */
  def onStmt(namespace: => Namespace)(stmt: Statement): Statement =
    stmt.map(onStmt(namespace)) match {
      // Proper union types would deduplicate this code
      case r: DefRegister if illegalClockExpr(r.clock) =>
        val node = DefNode(r.info, namespace.newTemp, r.clock)
        val rx = r.copy(clock = WRef(node))
        Block(Seq(node, rx))
      case p: Print if illegalClockExpr(p.clk) =>
        val node = DefNode(p.info, namespace.newTemp, p.clk)
        val px = p.copy(clk = WRef(node))
        Block(Seq(node, px))
      case s: Stop if illegalClockExpr(s.clk) =>
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

/** Ensure Clocks to be emitted are legal Verilog */
class LegalizeClocksTransform extends Transform with PreservesAll[Transform] {
  def inputForm = UnknownForm
  def outputForm = UnknownForm

  override val prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq( Dependency[BlackBoxSourceHelper],
         Dependency[FixAddingNegativeLiterals],
         Dependency[ReplaceTruncatingArithmetic],
         Dependency[InlineBitExtractionsTransform],
         Dependency[InlineCastsTransform] )

  override val optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override val dependents = Seq.empty

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map(LegalizeClocksTransform.onMod(_))
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
