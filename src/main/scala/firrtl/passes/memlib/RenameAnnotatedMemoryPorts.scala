// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import AnalysisUtils._
import MemPortUtils._
import MemTransformUtils._


/** Changes memory port names to standard port names (i.e. RW0 instead T_408)
 */
object RenameAnnotatedMemoryPorts extends Pass {
  /** Renames memory ports to a standard naming scheme:
   *    - R0, R1, ... for each read port
   *    - W0, W1, ... for each write port
   *    - RW0, RW1, ... for each readwrite port
   */
  def createMemProto(m: DefAnnotatedMemory): DefAnnotatedMemory = {
    val rports = m.readers.indices map (i => s"R$i")
    val wports = m.writers.indices map (i => s"W$i")
    val rwports = m.readwriters.indices map (i => s"RW$i")
    m copy (readers = rports, writers = wports, readwriters = rwports)
  }

  /** Maps the serialized form of all memory port field names to the
   *    corresponding new memory port field Expression.
   *  E.g.:
   *    - ("m.read.addr") becomes (m.R0.addr)
   */
  def getMemPortMap(m: DefAnnotatedMemory, memPortMap: MemPortMap) {
    val defaultFields = Seq("addr", "en", "clk")
    val rFields = defaultFields :+ "data"
    val wFields = rFields :+ "mask"
    val rwFields = defaultFields ++ Seq("wmode", "wdata", "rdata", "wmask")

    def updateMemPortMap(ports: Seq[String], fields: Seq[String], newPortKind: String): Unit =
      for ((p, i) <- ports.zipWithIndex; f <- fields) {
        val newPort = WSubField(WRef(m.name), newPortKind + i)
        val field = WSubField(newPort, f)
        memPortMap(s"${m.name}.$p.$f") = field
      }
    updateMemPortMap(m.readers, rFields, "R")
    updateMemPortMap(m.writers, wFields, "W")
    updateMemPortMap(m.readwriters, rwFields, "RW")
  }

  /** Replaces candidate memories with memories with standard port names
    * Does not update the references (this is done via updateStmtRefs)
    */
  def updateMemStmts(memPortMap: MemPortMap)(s: Statement): Statement = s match {
    case m: DefAnnotatedMemory =>
      val updatedMem = createMemProto(m)
      getMemPortMap(m, memPortMap)
      updatedMem
    case s => s map updateMemStmts(memPortMap)
  }

  /** Replaces candidate memories and their references with standard port names
   */
  def updateMemMods(m: DefModule) = {
    val memPortMap = new MemPortMap
    (m map updateMemStmts(memPortMap)
       map updateStmtRefs(memPortMap))
  }

  def run(c: Circuit) = c copy (modules = c.modules map updateMemMods)
}
