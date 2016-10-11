// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import AnalysisUtils._
import MemPortUtils._
import MemTransformUtils._

object MemTransformUtils {
  def getFillWMask(mem: DefMemory) =
    getInfo(mem.info, "maskGran") match {
      case None => false
      case Some(maskGran) => maskGran == 1
    }

  def rPortToBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) :+ Field("data", Flip, mem.dataType))
  def rPortToFlattenBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) :+ Field("data", Flip, flattenType(mem.dataType)))

  def wPortToBundle(mem: DefMemory) = BundleType(
    (defaultPortSeq(mem) :+ Field("data", Default, mem.dataType)) ++
    (if (!containsInfo(mem.info, "maskGran")) Nil
     else Seq(Field("mask", Default, createMask(mem.dataType))))
  )
  def wPortToFlattenBundle(mem: DefMemory) = BundleType(
    (defaultPortSeq(mem) :+ Field("data", Default, flattenType(mem.dataType))) ++
    (if (!containsInfo(mem.info, "maskGran")) Nil
     else if (getFillWMask(mem)) Seq(Field("mask", Default, flattenType(mem.dataType)))
     else Seq(Field("mask", Default, flattenType(createMask(mem.dataType)))))
  )
  // TODO: Don't use createMask???

  def rwPortToBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, BoolType),
      Field("wdata", Default, mem.dataType),
      Field("rdata", Flip, mem.dataType)
    ) ++ (if (!containsInfo(mem.info, "maskGran")) Nil
     else Seq(Field("wmask", Default, createMask(mem.dataType)))
    )
  )

  def rwPortToFlattenBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, BoolType),
      Field("wdata", Default, flattenType(mem.dataType)),
      Field("rdata", Flip, flattenType(mem.dataType))
    ) ++ (if (!containsInfo(mem.info, "maskGran")) Nil
     else if (getFillWMask(mem)) Seq(Field("wmask", Default, flattenType(mem.dataType)))
     else Seq(Field("wmask", Default, flattenType(createMask(mem.dataType))))
    )  
  )

  def memToBundle(s: DefMemory) = BundleType(
    s.readers.map(Field(_, Flip, rPortToBundle(s))) ++
    s.writers.map(Field(_, Flip, wPortToBundle(s))) ++
    s.readwriters.map(Field(_, Flip, rwPortToBundle(s))))
  
  def memToFlattenBundle(s: DefMemory) = BundleType(
    s.readers.map(Field(_, Flip, rPortToFlattenBundle(s))) ++
    s.writers.map(Field(_, Flip, wPortToFlattenBundle(s))) ++
    s.readwriters.map(Field(_, Flip, rwPortToFlattenBundle(s))))


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
        memPortMap(s"${m.name}.$p.$f") = field
      }
    updateMemPortMap(m.readers, rFields, "R")
    updateMemPortMap(m.writers, wFields, "W")
    updateMemPortMap(m.readwriters, rwFields, "RW")
    memPortMap
  }

  def createMemProto(m: DefMemory) = {
    val rports = m.readers.indices map (i => s"R$i")
    val wports = m.writers.indices map (i => s"W$i")
    val rwports = m.readwriters.indices map (i => s"RW$i")
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
        case sx => sx
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
    case sx => sx map updateMemStmts(uniqueMems, memPortMap)
  }

  def updateMemMods(m: DefModule) = {
    val uniqueMems = new Memories
    val memPortMap = new MemPortMap
    (m map updateMemStmts(uniqueMems, memPortMap)
       map updateStmtRefs(memPortMap))
  }

  def run(c: Circuit) = c copy (modules = c.modules map updateMemMods)
}
// TODO: Module namespace?
