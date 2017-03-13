// See LICENSE for license details.

package chisel3
package tutorial
package lesson1

import chisel3._
import chisel3.experimental.ChiselAnnotation

import logger._

import firrtl.annotations._
import firrtl.{Transform, LowForm, CircuitState}
import firrtl.ir.{Circuit, DefModule, Statement, Expression, Mux}
import firrtl.Mappers._

import scala.collection.mutable

trait AnalyzeModule {
  self: Module =>
  def setAnalyze(): Unit = {
    annotate(ChiselAnnotation(this, classOf[AnalyzeCircuit], ""))
  }
}

object AnalyzeAnnotation {
  def apply(target: ModuleName): Annotation = Annotation(target, classOf[AnalyzeCircuit], "")
  def unapply(a: Annotation): Option[ModuleName] = a match {
    case Annotation(ModuleName(n, c), x, "") if x == classOf[AnalyzeCircuit] =>
      Some(ModuleName(n, c))
    case _ =>
      None
  }
}

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
  * See firrtl.tutorial for more information
  */
class AnalyzeCircuit extends Transform with LazyLogging {
  def inputForm = LowForm
  def outputForm = LowForm

  // Called by [[Compiler]] to run your pass. [[CircuitState]] contains
  // the circuit and its form, as well as other related data.
  def execute(state: CircuitState): CircuitState =
    getMyAnnotations(state) match {
      case Nil => state
      case annos =>
        val modules = annos.collect { case AnalyzeAnnotation(ModuleName(m, c)) => m }
        state.copy(circuit = run(state.circuit, modules))
    }

  def run(circuit: Circuit, modules: Seq[String]): Circuit = {
    val ledger = new Ledger()

    // Execute the function walkModule(ledger) on every [[DefModule]] in
    // circuit, returning a new [[Circuit]] with new [[Seq]] of [[DefModule]].
    circuit.modules.filter { m =>
      modules.contains(m.name)
    }.map(walkModule(ledger))

    // Print our ledger
    logger.info(ledger.serialize)
    println(ledger.serialize)

    // Return an unchanged [[Circuit]]
    circuit
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

    // Execute the function walkExpression(ledger) on every [[Expression]] in s.
    s map walkExpression(ledger)

    // Execute the function walkStatement(ledger) on every [[Statement]] in s.
    s map walkStatement(ledger)
  }

  // Deeply visits every [[Expression]] in e.
  def walkExpression(ledger: Ledger)(e: Expression): Expression = {

    // Execute the function walkExpression(ledger) on every [[Expression]] in e.
    val visited = e map walkExpression(ledger)

    visited match {
      // If e is a [[Mux]], increment our ledger and return e.
      case Mux(cond, tval, fval, tpe) =>
        ledger.foundMux
        e
      // If e is not a [[Mux]], return e.
      case notmux => notmux
    }
  }
}
