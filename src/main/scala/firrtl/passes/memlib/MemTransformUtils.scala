// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import AnalysisUtils._
import MemPortUtils.{MemPortMap}

object MemTransformUtils {

  /** Replaces references to old memory port names with new memory port names
   */
  def updateStmtRefs(repl: MemPortMap)(s: Statement): Statement = {
    //TODO(izraelevitz): check speed
    def updateRef(e: Expression): Expression = {
      val ex = e map updateRef
      repl getOrElse (ex.serialize, ex)
    }

    def hasEmptyExpr(stmt: Statement): Boolean = {
      var foundEmpty = false
      def testEmptyExpr(e: Expression): Expression = {
        e match {
          case EmptyExpression => foundEmpty = true
          case _ =>
        }
        e map testEmptyExpr // map must return; no foreach
      }
      stmt map testEmptyExpr
      foundEmpty
    }

    def updateStmtRefs(s: Statement): Statement =
      s map updateStmtRefs map updateRef match {
        case c: Connect if hasEmptyExpr(c) => EmptyStmt
        case s => s
      }

    updateStmtRefs(s)
  }

  def defaultPortSeq(mem: DefAnnotatedMemory): Seq[Field] = MemPortUtils.defaultPortSeq(mem.toMem)
  def memPortField(s: DefAnnotatedMemory, p: String, f: String): Expression =
    MemPortUtils.memPortField(s.toMem, p, f)
}
