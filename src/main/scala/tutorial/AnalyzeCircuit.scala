package tutorial

// Compiler Infrastructure
import firrtl.{Transform, LowForm, CircuitState}
// Firrtl IR classes
import firrtl.ir.{Circuit, DefModule, Statement, Expression, Mux}
// Map functions
import firrtl.Mappers._
// Scala's mutable collections
import scala.collection.mutable

/** Ledger
  *
  * Use for tracking [[Circuit]] statistics
  * See [[AnalyzeCircuit]]
  */
class Ledger {
  private var moduleName: Option[String] = None
  private val moduleMuxMap = mutable.Map[String, Int]()
  def foundMux: Unit = moduleName match {
    case None => error("Module name not defined in Ledger!")
    case Some(name) => moduleMuxMap(name) = moduleMuxMap.getOrElse(name, 0) + 1
  }
  def setModuleName(name: String): Unit = {
    moduleName = Some(name)
  }
  def serialize: String = {
    moduleMuxMap map { case (module, nMux) => s"$module => $nMux muxes!" } mkString "\n"
  }
}

/** AnalyzeCircuit Transform
  *
  * Walks [[ir.Circuit]], and records the number of muxes it finds, per module.
  *
  * See the following links for more detailed explanations:
  * Firrtl's IR:
  *   - https://github.com/ucb-bar/firrtl/wiki/Understanding-Firrtl-Intermediate-Representation
  * Traversing a circuit:
  *   - https://github.com/ucb-bar/firrtl/wiki/traversing-a-circuit for more
  * Common Pass Idioms:
  *   - https://github.com/ucb-bar/firrtl/wiki/Common-Pass-Idioms
  */
class AnalyzeCircuit extends Transform {
  // Requires the [[Circuit]] form to be "low"
  def inputForm = LowForm
  // Indicates the output [[Circuit]] form to be "low"
  def outputForm = LowForm

  // Called by [[Compiler]] to run your pass. [[CircuitState]] contains
  // the circuit and its form, as well as other related data.
  def execute(state: CircuitState): CircuitState = {
    val ledger = new Ledger()
    val circuit = state.circuit

    // Execute the function walkModule(ledger) on every [[DefModule]] in
    // circuit, returning a new [[Circuit]] with new [[Seq]] of [[DefModule]].
    //   - "higher order functions" - using a function as an object
    //   - "function currying" - partial argument notation
    //   - "infix notation" - fancy function calling syntax
    //   - "map" - classic functional programming concept
    //   - discard the returned new [[Circuit]] because circuit is unmodified
    circuit map walkModule(ledger)

    // Print our ledger
    println(ledger.serialize)

    // Return an unchanged [[CircuitState]]
    state
  }

  // Deeply visits every [[Statement]] in m.
  def walkModule(ledger: Ledger)(m: DefModule): DefModule = {
    // Set ledger to current module name
    ledger.setModuleName(m.name)

    // Execute the function walkStatement(ledger) on every [[Statement]] in m.
    //   - return the new [[DefModule]] (in this case, its identical to m)
    //   - if m does not contain [[Statement]], map returns m.
    m map walkStatement(ledger)
  }

  // Deeply visits every [[Statement]] and [[Expression]] in s.
  def walkStatement(ledger: Ledger)(s: Statement): Statement = {

    // Execute the function walkExpression(ledger) on every [[Expression]] in s.
    //   - discard the new [[Statement]] (in this case, its identical to s)
    //   - if s does not contain [[Expression]], map returns s.
    s map walkExpression(ledger)

    // Execute the function walkStatement(ledger) on every [[Statement]] in s.
    //   - return the new [[Statement]] (in this case, its identical to s)
    //   - if s does not contain [[Statement]], map returns s.
    s map walkStatement(ledger)
  }

  // Deeply visits every [[Expression]] in e.
  //   - "post-order traversal" - handle e's children [[Expression]] before e
  def walkExpression(ledger: Ledger)(e: Expression): Expression = {

    // Execute the function walkExpression(ledger) on every [[Expression]] in e.
    //   - return the new [[Expression]] (in this case, its identical to e)
    //   - if s does not contain [[Expression]], map returns e.
    val visited = e map walkExpression(ledger)

    visited match {
      // If e is a [[Mux]], increment our ledger and return e.
      case Mux(cond, tval, fval, tpe) =>
        ledger.foundMux
        e
      // If e is not a [[Mux]], return e.
      case e => e
    }
  }
}
