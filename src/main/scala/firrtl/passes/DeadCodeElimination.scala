// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._

import annotation.tailrec

object DeadCodeElimination extends Pass {
  def name = "Dead Code Elimination"

  private def dceOnce(s: Statement): (Statement, Long) = {
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
  private def dce(s: Statement): Statement = {
    val (res, n) = dceOnce(s)
    if (n > 0) dce(res) else res
  }

  def run(c: Circuit): Circuit = {
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module => Module(m.info, m.name, m.ports, dce(m.body))
    }
    Circuit(c.info, modulesx, c.main)
  }
}
