// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._

import annotation.tailrec

object CommonSubexpressionElimination extends Pass {
  def name = "Common Subexpression Elimination"

  private def cseOnce(s: Statement): (Statement, Long) = {
    var nEliminated = 0L
    val expressions = collection.mutable.HashMap[MemoizedHash[Expression], String]()
    val nodes = collection.mutable.HashMap[String, Expression]()

    def recordNodes(s: Statement): Statement = s match {
      case x: DefNode =>
        nodes(x.name) = x.value
        expressions.getOrElseUpdate(x.value, x.name)
        x
      case _ => s map recordNodes
    }

    def eliminateNodeRef(e: Expression): Expression = e match {
      case WRef(name, tpe, kind, gender) => nodes get name match {
        case Some(expression) => expressions get expression match {
          case Some(cseName) if cseName != name =>
            nEliminated += 1
            WRef(cseName, tpe, kind, gender)
          case _ => e
        }
        case _ => e
      }
      case _ => e map eliminateNodeRef
    }

    def eliminateNodeRefs(s: Statement): Statement = s map eliminateNodeRefs map eliminateNodeRef

    recordNodes(s)
    (eliminateNodeRefs(s), nEliminated)
  }

  @tailrec
  private def cse(s: Statement): Statement = {
    val (res, n) = cseOnce(s)
    if (n > 0) cse(res) else res
  }

  def run(c: Circuit): Circuit = {
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module => Module(m.info, m.name, m.ports, cse(m.body))
    }
    Circuit(c.info, modulesx, c.main)
  }
}
