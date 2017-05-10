// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._

import annotation.tailrec

object DeadCodeElimination extends Transform {
  def inputForm = UnknownForm
  def outputForm = UnknownForm
  private def dceOnce(renames: RenameMap)(s: Statement): (Statement, Long) = {
    val referenced = collection.mutable.HashSet[String]()
    var nEliminated = 0L

    def checkExpressionUse(e: Expression): Expression = {
      e match {
        case WRef(name, _, _, _) => referenced += name
        case _ => e map checkExpressionUse
      }
      e
    }

    def checkUse(s: Statement): Statement = s map checkUse map checkExpressionUse

    def maybeEliminate(x: Statement, name: String) =
      if (referenced(name)) x
      else {
        nEliminated += 1
        renames.delete(name)
        EmptyStmt
      }

    def removeUnused(s: Statement): Statement = s match {
      case x: DefRegister => maybeEliminate(x, x.name)
      case x: DefWire => maybeEliminate(x, x.name)
      case x: DefNode => maybeEliminate(x, x.name)
      case x => s map removeUnused
    }

    checkUse(s)
    (removeUnused(s), nEliminated)
  }

  @tailrec
  private def dce(renames: RenameMap)(s: Statement): Statement = {
    val (res, n) = dceOnce(renames)(s)
    if (n > 0) dce(renames)(res) else res
  }

  def execute(state: CircuitState): CircuitState = {
    val c = state.circuit
    val renames = RenameMap()
    renames.setCircuit(c.main)
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module =>
        renames.setModule(m.name)
        Module(m.info, m.name, m.ports, dce(renames)(m.body))
    }
    val result = Circuit(c.info, modulesx, c.main)
    CircuitState(result, outputForm, state.annotations, Some(renames))
  }
}
