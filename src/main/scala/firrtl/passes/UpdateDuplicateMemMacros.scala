// See LICENSE for license details.

package firrtl.passes

import scala.collection.mutable
import AnalysisUtils._
import MemTransformUtils._
import firrtl.ir._
import firrtl._
import firrtl.Mappers._
import firrtl.Utils._

object MemTransformUtils {

  def createRef(n: String) = WRef(n, UnknownType, ExpKind(), UNKNOWNGENDER)
  def createSubField(exp: Expression, n: String) = WSubField(exp, n, UnknownType, UNKNOWNGENDER)
  def connectFields(lref: Expression, lname: String, rref: Expression, rname: String) = 
    Connect(NoInfo, createSubField(lref, lname), createSubField(rref, rname))

  def getMemPortMap(m: DefMemory) = {
    val memPortMap = mutable.HashMap[String, Expression]()
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
    memPortMap.toMap
  }
  def createMemProto(m: DefMemory) = {
    val rports = (0 until m.readers.length) map (i => s"R$i")
    val wports = (0 until m.writers.length) map (i => s"W$i")
    val rwports = (0 until m.readwriters.length) map (i => s"RW$i")
    m.copy(readers = rports, writers = wports, readwriters = rwports)
  }

  def updateStmtRefs(s: Statement, repl: Map[String, Expression]): Statement = {
    def updateRef(e: Expression): Expression = e map updateRef match {
      case e: WSubField => repl getOrElse (e.serialize, e)
      case e => e
    }
    def updateStmtRefs(s: Statement): Statement = s map updateStmtRefs map updateRef match {
      case Connect(info, EmptyExpression, exp) => EmptyStmt 
      case Connect(info, WSubIndex(EmptyExpression, _, _, _), exp)  => EmptyStmt
      case s => s
    }
    updateStmtRefs(s)
  }

}

object UpdateDuplicateMemMacros extends Pass {

  def name = "Convert memory port names to be more meaningful and tag duplicate memories"

  def run(c: Circuit) = {
    val uniqueMems = mutable.ArrayBuffer[DefMemory]()

    def updateMemMods(m: Module) = {
      val memPortMap = mutable.HashMap[String, Expression]()

      def updateMemStmts(s: Statement): Statement = s match {
        case m: DefMemory if containsInfo(m.info, "useMacro") => 
          val updatedMem = createMemProto(m)
          memPortMap ++= getMemPortMap(m)
          val proto = uniqueMems find (x => eqMems(x, updatedMem))
          if (proto == None) {
            uniqueMems += updatedMem
            updatedMem
          } 
          else updatedMem.copy(info = appendInfo(updatedMem.info, "ref" -> proto.get.name))
        case b: Block => b map updateMemStmts
        case s => s
      }

      val updatedMems = updateMemStmts(m.body)
      val updatedConns = updateStmtRefs(updatedMems, memPortMap.toMap)
      m.copy(body = updatedConns)
    }

    val updatedMods = c.modules map {
      case m: Module => updateMemMods(m)
      case m: ExtModule => m
    }
    c.copy(modules = updatedMods) 
  }  

}
// TODO: Module namespace?
