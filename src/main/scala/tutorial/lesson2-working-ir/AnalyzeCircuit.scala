// See LICENSE for license details.

package tutorial
package lesson2

// Compiler Infrastructure
import firrtl.{Transform, LowForm, CircuitState, Utils}
// Firrtl IR classes
import firrtl.ir.{Circuit, DefModule, Statement, DefInstance, Expression, Mux}
// Firrtl compiler's working IR classes (WIR)
import firrtl.WDefInstance
// Map functions
import firrtl.Mappers._
// Scala's mutable collections
import scala.collection.mutable

/** Ledger tracks [[firrtl.ir.Circuit]] statistics
  * 
  * In this lesson, we want to calculate the number of muxes, not just in 
  *  a module, but also in any instances it has of other modules, etc.
  *
  * To do this, we need to update our Ledger class to keep track of this
  *  module instance information
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
  * Walks [[firrtl.ir.Circuit]], and records the number of muxes and instances it
  *  finds, per module.
  *
  * While the Firrtl parser emits a bare form of the IR (located in firrtl.ir._),
  *  it is often useful to have more information in these case classes. To do this,
  *  the Firrtl compiler has mirror "working" classes for the following IR
  *  nodes (which contain additional fields):
  *   - DefInstance -> WDefInstance
  *   - SubAccess -> WSubAccess
  *   - SubIndex -> WSubIndex
  *   - SubField -> WSubField
  *   - Reference -> WRef
  *
  * Take a look at [[ToWorkingIR]] in src/main/scala/firrtl/passes/Passes.scala
  *  to see how Firrtl IR nodes are replaced with working IR nodes.
  *
  * Future lessons will explain the WIR's additional fields. For now, it is
  *  enough to know that the transform [[ResolveAndCheck]] populates these
  *  fields, and checks the legality of the circuit. If your transform is
  *  creating new WIR nodes, use the following "unknown" values in the WIR
  *  node, and then call [[ResolveAndCheck]] at the end of your transform:
  *   - Kind -> ExpKind
  *   - Gender -> UNKNOWNGENDER
  *   - Type -> UnknownType
  *
  * The following [[CircuitForm]]s require WIR instead of IR nodes:
  *   - HighForm
  *   - MidForm
  *   - LowForm
  *
  * See the following links for more detailed explanations:
  * IR vs Working IR
  *   - TODO(izraelevitz) 
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
      // IR node [[DefInstance]] is previously replaced by WDefInstance, a
      //  "working" IR node
      case DefInstance(info, name, module) => 
        Utils.error("All DefInstances should have been replaced by WDefInstances")
      // Working IR Node [[WDefInstance]] is what the compiler uses
      // See src/main/scala/firrtl/WIR.scala for all working IR nodes
      case WDefInstance(info, name, module, tpe) =>
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
