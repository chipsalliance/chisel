// See LICENSE for license details.

package tutorial
package lesson2

// Compiler Infrastructure
import firrtl.{Transform, LowForm, CircuitState}
// Firrtl IR classes
import firrtl.ir.{DefModule, Statement, Expression, Mux, DefInstance}
// Map functions
import firrtl.Mappers._
// Scala's mutable collections
import scala.collection.mutable

/** Ledger tracks [[firrtl.ir.Circuit]] statistics
  *
  * In this lesson, we want to calculate the number of muxes, not just in a module, but also in any instances it has of
  * other modules, etc.
  *
  * To do this, we need to update our Ledger class to keep track of this module instance information
  *
  * See [[lesson2.AnalyzeCircuit]]
  */
class Ledger {
  private var moduleName: Option[String] = None
  private val modules = mutable.Set[String]()
  private val moduleMuxMap = mutable.Map[String, Int]()
  private val moduleInstanceMap = mutable.Map[String, Seq[String]]()
  def getModuleName: String = moduleName match {
    case None => sys.error("Module name not defined in Ledger!")
    case Some(name) => name
  }
  def setModuleName(myName: String): Unit = {
    modules += myName
    moduleName = Some(myName)
  }
  def foundMux(): Unit = {
    val myName = getModuleName
    moduleMuxMap(myName) = moduleMuxMap.getOrElse(myName, 0) + 1
  }
  // Added this function to track when a module instantiates another module
  def foundInstance(name: String): Unit = {
    val myName = getModuleName
    moduleInstanceMap(myName) = moduleInstanceMap.getOrElse(myName, Nil) :+ name
  }
  // Counts mux's in a module, and all its instances (recursively).
  private def countMux(myName: String): Int = {
    val myMuxes = moduleMuxMap.getOrElse(myName, 0)
    val myInstanceMuxes =
      moduleInstanceMap.getOrElse(myName, Nil).foldLeft(0) {
        (total, name) => total + countMux(name)
      }
    myMuxes + myInstanceMuxes
  }
  // Display recursive total of muxes
  def serialize: String = {
    modules map { myName => s"$myName => ${countMux(myName)} muxes!" } mkString "\n"
  }
}

/** AnalyzeCircuit Transform
  *
  * Walks [[firrtl.ir.Circuit]], and records the number of muxes and instances it finds, per module.
  *
  * While the Firrtl IR specification describes a written format, the AST nodes used internally by the
  * implementation have additional "analysis" fields.
  *
  * Take a look at [[firrtl.ir.Reference Reference]] to see how a reference to a component name is
  * augmented with relevant type, kind (memory, wire, etc), and flow information.
  *
  * Future lessons will explain the IR's additional fields. For now, it is enough to know that declaring
  * [[firrtl.stage.Forms.Resolved]] as a prerequisite is a handy shorthand for ensuring that all of these
  * fields will be populated with accurant information before your transform runs. If you create new IR
  * nodes and do not wish to calculate the proper final values for all these fields, you can populate them
  * with default 'unknown' values.
  *   - Kind -> ExpKind
  *   - Flow -> UnknownFlow
  *   - Type -> UnknownType
  *
  */
class AnalyzeCircuit extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm

  // Called by [[Compiler]] to run your pass.
  def execute(state: CircuitState): CircuitState = {
    val ledger = new Ledger()
    val circuit = state.circuit

    // Execute the function walkModule(ledger) on all [[DefModule]] in circuit
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
    m map walkStatement(ledger)
  }

  // Deeply visits every [[Statement]] and [[Expression]] in s.
  def walkStatement(ledger: Ledger)(s: Statement): Statement = {
    // Map the functions walkStatement(ledger) and walkExpression(ledger)
    val visited = s map walkStatement(ledger) map walkExpression(ledger)
    visited match {
      case DefInstance(info, name, module, tpe) =>
        ledger.foundInstance(module)
        visited
      case _ => visited
    }
  }

  // Deeply visits every [[Expression]] in e.
  def walkExpression(ledger: Ledger)(e: Expression): Expression = {
    // Execute the function walkExpression(ledger) on every [[Expression]] in e,
    //  then handle if a [[Mux]].
    e map walkExpression(ledger) match {
      case mux: Mux =>
        ledger.foundMux()
        mux
      case notmux => notmux
    }
  }
}
