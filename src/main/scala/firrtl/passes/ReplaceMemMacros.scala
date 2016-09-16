// See LICENSE for license details.

package firrtl.passes

import scala.collection.mutable
import firrtl.ir._
import AnalysisUtils._
import MemTransformUtils._
import firrtl._
import firrtl.Utils._
import MemPortUtils._
import firrtl.Mappers._

class ReplaceMemMacros(writer: ConfWriter) extends Pass {

  def name = "Replace memories with black box wrappers (optimizes when write mask isn't needed) + configuration file"

  def run(c: Circuit) = {

    lazy val moduleNamespace = Namespace(c)
    val memMods = mutable.ArrayBuffer[DefModule]()
    val uniqueMems = mutable.ArrayBuffer[DefMemory]()

    def updateMemMods(m: Module) = {
      val memPortMap = new MemPortMap

      def updateMemStmts(s: Statement): Statement = s match {
        case m: DefMemory if containsInfo(m.info, "useMacro") => 
          if(!containsInfo(m.info, "maskGran")) {
            m.writers foreach { w => memPortMap(s"${m.name}.${w}.mask") = EmptyExpression }
            m.readwriters foreach { w => memPortMap(s"${m.name}.${w}.wmask") = EmptyExpression }
          }
          val infoT = getInfo(m.info, "info")
          val info = if (infoT == None) NoInfo else infoT.get match { case i: Info => i }
          val ref = getInfo(m.info, "ref")
          
          // prototype mem
          if (ref == None) {
            val newWrapperName = moduleNamespace.newName(m.name)
            val newMemBBName = moduleNamespace.newName(m.name + "_ext")
            val newMem = m.copy(name = newMemBBName)
            memMods ++= createMemModule(newMem, newWrapperName)
            uniqueMems += newMem
            WDefInstance(info, m.name, newWrapperName, UnknownType) 
          }
          else {
            val r = ref.get match { case s: String => s }
            WDefInstance(info, m.name, r, UnknownType) 
          }
        case b: Block => b map updateMemStmts
        case s => s
      }

      val updatedMems = updateMemStmts(m.body)
      val updatedConns = updateStmtRefs(memPortMap)(updatedMems)
      m.copy(body = updatedConns)
    }

    val updatedMods = c.modules map {
      case m: Module => updateMemMods(m)
      case m: ExtModule => m
    }

    // print conf
    writer.serialize
    c.copy(modules = updatedMods ++ memMods.toSeq) 
  }  

  // from Albert
  def createMemModule(m: DefMemory, wrapperName: String): Seq[DefModule] = {
    assert(m.dataType != UnknownType)
    val stmts = mutable.ArrayBuffer[Statement]()
    val wrapperioPorts = MemPortUtils.memToBundle(m).fields.map(f => Port(NoInfo, f.name, Input, f.tpe)) 
    val bbProto = m.copy(dataType = flattenType(m.dataType))
    val bbioPorts = MemPortUtils.memToFlattenBundle(m).fields.map(f => Port(NoInfo, f.name, Input, f.tpe)) 

    stmts += WDefInstance(NoInfo, m.name, m.name, UnknownType)
    val bbRef = createRef(m.name)
    stmts ++= (m.readers zip bbProto.readers).flatMap { 
      case (x, y) => adaptReader(createRef(x), m, createSubField(bbRef, y), bbProto)
    }
    stmts ++= (m.writers zip bbProto.writers).flatMap { 
      case (x, y) => adaptWriter(createRef(x), m, createSubField(bbRef, y), bbProto)
    }
    stmts ++= (m.readwriters zip bbProto.readwriters).flatMap { 
      case (x, y) => adaptReadWriter(createRef(x), m, createSubField(bbRef, y), bbProto)
    } 
    val wrapper = Module(NoInfo, wrapperName, wrapperioPorts, Block(stmts))   
    val bb = ExtModule(NoInfo, m.name, bbioPorts) 
    // TODO: Annotate? -- use actual annotation map

    // add to conf file
    writer.append(m)
    Seq(bb, wrapper)
  }

  // TODO: get rid of copy pasta
  def adaptReader(wrapperPort: Expression, wrapperMem: DefMemory, bbPort: Expression, bbMem: DefMemory) = Seq(
    connectFields(bbPort, "addr", wrapperPort, "addr"),
    connectFields(bbPort, "en", wrapperPort, "en"),
    connectFields(bbPort, "clk", wrapperPort, "clk"),
    fromBits(
      WSubField(wrapperPort, "data", wrapperMem.dataType, UNKNOWNGENDER),
      WSubField(bbPort, "data", bbMem.dataType, UNKNOWNGENDER)
    )
  )

  def adaptWriter(wrapperPort: Expression, wrapperMem: DefMemory, bbPort: Expression, bbMem: DefMemory) = {
    val defaultSeq = Seq(
      connectFields(bbPort, "addr", wrapperPort, "addr"),
      connectFields(bbPort, "en", wrapperPort, "en"),
      connectFields(bbPort, "clk", wrapperPort, "clk"),
      Connect(
        NoInfo,
        WSubField(bbPort, "data", bbMem.dataType, UNKNOWNGENDER),
        toBits(WSubField(wrapperPort, "data", wrapperMem.dataType, UNKNOWNGENDER))
      )
    )
    if (containsInfo(wrapperMem.info, "maskGran")) {
      val wrapperMask = createMask(wrapperMem.dataType)
      val fillWMask = getFillWMask(wrapperMem)
      val bbMask = if (fillWMask) flattenType(wrapperMem.dataType) else flattenType(wrapperMask)
      val rhs = {
        if (fillWMask) toBitMask(WSubField(wrapperPort, "mask", wrapperMask, UNKNOWNGENDER), wrapperMem.dataType)
        else toBits(WSubField(wrapperPort, "mask", wrapperMask, UNKNOWNGENDER))
      }
      defaultSeq :+ Connect(
        NoInfo,
        WSubField(bbPort, "mask", bbMask, UNKNOWNGENDER),
        rhs
      )
    }
    else defaultSeq
  }

  def adaptReadWriter(wrapperPort: Expression, wrapperMem: DefMemory, bbPort: Expression, bbMem: DefMemory) = {
    val defaultSeq = Seq(
      connectFields(bbPort, "addr", wrapperPort, "addr"),
      connectFields(bbPort, "en", wrapperPort, "en"),
      connectFields(bbPort, "clk", wrapperPort, "clk"),
      connectFields(bbPort, "wmode", wrapperPort, "wmode"),
      Connect(
        NoInfo,
        WSubField(bbPort, "wdata", bbMem.dataType, UNKNOWNGENDER),
        toBits(WSubField(wrapperPort, "wdata", wrapperMem.dataType, UNKNOWNGENDER))
      ),
      fromBits(
        WSubField(wrapperPort, "rdata", wrapperMem.dataType, UNKNOWNGENDER),
        WSubField(bbPort, "rdata", bbMem.dataType, UNKNOWNGENDER)
      )
    )
    if (containsInfo(wrapperMem.info, "maskGran")) {
      val wrapperMask = createMask(wrapperMem.dataType)
      val fillWMask = getFillWMask(wrapperMem)
      val bbMask = if (fillWMask) flattenType(wrapperMem.dataType) else flattenType(wrapperMask)
      val rhs = {
        if (fillWMask) toBitMask(WSubField(wrapperPort, "wmask", wrapperMask, UNKNOWNGENDER), wrapperMem.dataType)
        else toBits(WSubField(wrapperPort, "wmask", wrapperMask, UNKNOWNGENDER))
      }
      defaultSeq :+ Connect(
        NoInfo,
        WSubField(bbPort, "wmask", bbMask, UNKNOWNGENDER),
        rhs
      )
    }
    else defaultSeq
  }

}
