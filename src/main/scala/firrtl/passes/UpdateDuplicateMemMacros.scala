// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import AnalysisUtils._
import MemTransformUtils._

object MemTransformUtils {

  type MemPortMap = collection.mutable.HashMap[String, Expression]
  type Memories = collection.mutable.ArrayBuffer[DefMemory]

  def createRef(n: String) = WRef(n, UnknownType, ExpKind, UNKNOWNGENDER)
  def createSubField(exp: Expression, n: String) = WSubField(exp, n, UnknownType, UNKNOWNGENDER)
  def connectFields(lref: Expression, lname: String, rref: Expression, rname: String) = 
    Connect(NoInfo, createSubField(lref, lname), createSubField(rref, rname))

  def getMemPortMap(m: DefMemory) = {
    val memPortMap = new MemPortMap
    val defaultFields = Seq("addr", "en", "clk")
    val rFields = defaultFields :+ "data"
    val wFields = rFields :+ "mask"
    val rwFields = defaultFields ++ Seq("wmode", "wdata", "rdata", "wmask")

    def updateMemPortMap(ports: Seq[String], fields: Seq[String], portType: String) = 
      for ((p, i) <- ports.zipWithIndex; f <- fields) {
        val newPort = createSubField(createRef(m.name), portType+i)        
        val field = createSubField(newPort, f)
        memPortMap(s"${m.name}.${p}.${f}") = field
      }
    updateMemPortMap(m.readers, rFields, "R")
    updateMemPortMap(m.writers, wFields, "W")
    updateMemPortMap(m.readwriters, rwFields, "RW")
    memPortMap
  }

  def createMemProto(m: DefMemory) = {
    val rports = (0 until m.readers.length) map (i => s"R$i")
    val wports = (0 until m.writers.length) map (i => s"W$i")
    val rwports = (0 until m.readwriters.length) map (i => s"RW$i")
    m copy (readers = rports, writers = wports, readwriters = rwports)
  }

  def updateStmtRefs(repl: MemPortMap)(s: Statement): Statement = {
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

}

object UpdateDuplicateMemMacros extends Pass {

  def name = "Convert memory port names to be more meaningful and tag duplicate memories"

  def updateMemStmts(uniqueMems: Memories,
                     memPortMap: MemPortMap)
                     (s: Statement): Statement = s match {
    case m: DefMemory if containsInfo(m.info, "useMacro") => 
      val updatedMem = createMemProto(m)
      memPortMap ++= getMemPortMap(m)
      uniqueMems find (x => eqMems(x, updatedMem)) match {
        case None =>
          uniqueMems += updatedMem
          updatedMem
        case Some(proto) =>
          updatedMem copy (info = appendInfo(updatedMem.info, "ref" -> proto.name))
      }
    case s => s map updateMemStmts(uniqueMems, memPortMap)
  }

  def updateMemMods(m: DefModule) = {
    val uniqueMems = new Memories
    val memPortMap = new MemPortMap
    (m map updateMemStmts(uniqueMems, memPortMap)
       map updateStmtRefs(memPortMap))
  }

  def run(c: Circuit) = c copy (modules = (c.modules map updateMemMods)) 
}
// TODO: Module namespace?
