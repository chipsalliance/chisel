package tutorial

// Compiler Infrastructure
import firrtl.{Transform, LowForm, CircuitState}
// Firrtl IR classes
import firrtl.ir.{Circuit, DefModule, Statement, Expression, Mux}
// Map functions
import firrtl.Mappers._
// Scala's mutable collections
import scala.collection.mutable

class Ledger {
  var moduleName: Option[String] = None
  val moduleMuxMap = mutable.Map[String, Int]()
  def foundMux: Unit = moduleName match {
    case None => error("Module name not defined in Ledger!")
    case Some(name) => moduleMuxMap(name) = moduleMuxMap.getOrElse(name, 0) + 1
  }
  def setModuleName(name: String): Unit = {
    moduleName = Some(name)
  }
  def serialize: String = {
    moduleMuxMap map { case (module, nMux) => s"$module => $nMux" } mkString "\n"
  }
}

class AnalyzeCircuit extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm
  def execute(state: CircuitState): CircuitState = {
    val ledger = new Ledger()
    state.circuit map walkModule(ledger)
    println(ledger.serialize)
    state
  }
  def walkModule(ledger: Ledger)(m: DefModule): DefModule = {
    ledger.setModuleName(m.name)
    m map walkStatement(ledger)
  }
  def walkStatement(ledger: Ledger)(s: Statement): Statement = {
    s map walkExpression(ledger) map walkStatement(ledger)
  }
  def walkExpression(ledger: Ledger)(e: Expression): Expression = e match {
    case Mux(cond, tval, fval, tpe) =>
      ledger.foundMux
      e
    case e => e map walkExpression(ledger)
  }
}
