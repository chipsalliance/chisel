// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import MemPortUtils._
import MemTransformUtils._
import AnalysisUtils._

class ReplaceMemMacros(writer: ConfWriter) extends Pass {
  def name = "Replace memories with black box wrappers" +
             " (optimizes when write mask isn't needed) + configuration file"

  // from Albert
  def createMemModule(m: DefMemory, wrapperName: String): Seq[DefModule] = {
    assert(m.dataType != UnknownType)
    val wrapperIoType = memToBundle(m)
    val wrapperIoPorts = wrapperIoType.fields map (f => Port(NoInfo, f.name, Input, f.tpe))
    val bbIoType = memToFlattenBundle(m)
    val bbIoPorts = bbIoType.fields map (f => Port(NoInfo, f.name, Input, f.tpe))
    val bbRef = createRef(m.name, bbIoType)
    val hasMask = containsInfo(m.info, "maskGran")
    val fillMask = getFillWMask(m)
    def portRef(p: String) = createRef(p, field_type(wrapperIoType, p))
    val stmts = Seq(WDefInstance(NoInfo, m.name, m.name, UnknownType)) ++
      (m.readers flatMap (r => adaptReader(portRef(r), createSubField(bbRef, r)))) ++
      (m.writers flatMap (w => adaptWriter(portRef(w), createSubField(bbRef, w), hasMask, fillMask))) ++
      (m.readwriters flatMap (rw => adaptReadWriter(portRef(rw), createSubField(bbRef, rw), hasMask, fillMask)))
    val wrapper = Module(NoInfo, wrapperName, wrapperIoPorts, Block(stmts))
    val bb = ExtModule(NoInfo, m.name, bbIoPorts)
    // TODO: Annotate? -- use actual annotation map

    // add to conf file
    writer.append(m)
    Seq(bb, wrapper)
  }

  // TODO: get rid of copy pasta
  def defaultConnects(wrapperPort: WRef, bbPort: WSubField) =
    Seq("clk", "en", "addr") map (f => connectFields(bbPort, f, wrapperPort, f))

  def maskBits(mask: WSubField, dataType: Type, fillMask: Boolean) =
    if (fillMask) toBitMask(mask, dataType) else toBits(mask)

  def adaptReader(wrapperPort: WRef, bbPort: WSubField) =
    defaultConnects(wrapperPort, bbPort) :+
    fromBits(createSubField(wrapperPort, "data"), createSubField(bbPort, "data"))

  def adaptWriter(wrapperPort: WRef, bbPort: WSubField, hasMask: Boolean, fillMask: Boolean) = {
    val wrapperData = createSubField(wrapperPort, "data")
    val defaultSeq = defaultConnects(wrapperPort, bbPort) :+
      Connect(NoInfo, createSubField(bbPort, "data"), toBits(wrapperData))
    hasMask match {
      case false => defaultSeq
      case true => defaultSeq :+ Connect(
        NoInfo,
        createSubField(bbPort, "mask"),
        maskBits(createSubField(wrapperPort, "mask"), wrapperData.tpe, fillMask)
      )
    }
  }

  def adaptReadWriter(wrapperPort: WRef, bbPort: WSubField, hasMask: Boolean, fillMask: Boolean) = {
    val wrapperWData = createSubField(wrapperPort, "wdata")
    val defaultSeq = defaultConnects(wrapperPort, bbPort) ++ Seq(
      fromBits(createSubField(wrapperPort, "rdata"), createSubField(bbPort, "rdata")),
      connectFields(bbPort, "wmode", wrapperPort, "wmode"), 
      Connect(NoInfo, createSubField(bbPort, "wdata"), toBits(wrapperWData)))
    hasMask match {
      case false => defaultSeq
      case true => defaultSeq :+ Connect(
        NoInfo,
        createSubField(bbPort, "wmask"),
        maskBits(createSubField(wrapperPort, "wmask"), wrapperWData.tpe, fillMask)
      )
    }
  }

  def updateMemStmts(namespace: Namespace,
                     memPortMap: MemPortMap,
                     memMods: Modules)
                     (s: Statement): Statement = s match {
    case m: DefMemory if containsInfo(m.info, "useMacro") => 
      if (!containsInfo(m.info, "maskGran")) {
        m.writers foreach { w => memPortMap(s"${m.name}.${w}.mask") = EmptyExpression }
        m.readwriters foreach { w => memPortMap(s"${m.name}.${w}.wmask") = EmptyExpression }
      }
      val info = getInfo(m.info, "info") match {
        case None => NoInfo
        case Some(p: Info) => p
      }
      getInfo(m.info, "ref") match {
        case None =>
          // prototype mem
          val newWrapperName = namespace newName m.name
          val newMemBBName = namespace newName s"${m.name}_ext"
          val newMem = m copy (name = newMemBBName)
          memMods ++= createMemModule(newMem, newWrapperName)
          WDefInstance(info, m.name, newWrapperName, UnknownType) 
        case Some(ref: String) =>
          WDefInstance(info, m.name, ref, UnknownType) 
      }
    case s => s map updateMemStmts(namespace, memPortMap, memMods)
  }

  def updateMemMods(namespace: Namespace, memMods: Modules)(m: DefModule) = {
    val memPortMap = new MemPortMap

    (m map updateMemStmts(namespace, memPortMap, memMods)
       map updateStmtRefs(memPortMap))
  }

  def run(c: Circuit) = {
    val namespace = Namespace(c)
    val memMods = new Modules
    val modules = c.modules map updateMemMods(namespace, memMods)
    // print conf
    writer.serialize()
    c copy (modules = modules ++ memMods)
  }  
}
