package firrtl.passes

import scala.collection.mutable.{HashMap,ArrayBuffer}
import firrtl.ir._
import AnalysisUtils._
import MemTransformUtils._
import firrtl._
import firrtl.Utils._

object ReplaceMemMacros extends Pass {

  def name = "Replace memories with black box wrappers (optimizes when write mask isn't needed)"

  def run(c: Circuit) = {

    lazy val moduleNamespace = Namespace(c)
    val memMods = ArrayBuffer[DefModule]()
    val uniqueMems = ArrayBuffer[DefMemory]()

    def updateMemMods(m: Module) = {
      val memPortMap = HashMap[String,Expression]()

      def updateMemStmts(s: Statement): Statement = s match {
        case m: DefMemory if containsInfo(m.info,"useMacro") => 
          if(!containsInfo(m.info,"maskGran")) {
            m.writers foreach {w => memPortMap(s"${m.name}.${w}.mask") = EmptyExpression}
            m.readwriters foreach {w => memPortMap(s"${m.name}.${w}.wmask") = EmptyExpression}
          }
          val infoT = getInfo(m.info,"info")
          val info = if (infoT == None) NoInfo else infoT.get match {case i: Info => i}
          val ref = getInfo(m.info,"ref")
          
          // prototype mem
          if (ref == None) {
            val newName = moduleNamespace.newName(m.name)
            val newMem = m.copy(name = newName)
            memMods ++= createMemModule(newMem)
            uniqueMems += newMem
            WDefInstance(info, m.name, newMem.name, UnknownType) 
          }
          else {
            val r = ref.get match {case s: String => s}
            WDefInstance(info, m.name, r, UnknownType) 
          }
        case b: Block => Block(b.stmts map updateMemStmts)
        case s => s
      }

      val updatedMems = updateMemStmts(m.body)
      val updatedConns = updateStmtRefs(updatedMems,memPortMap.toMap)
      m.copy(body = updatedConns)
    }

    val updatedMods = c.modules map {
      case m: Module => updateMemMods(m)
      case m: ExtModule => m
    }

    c.copy(modules = updatedMods ++ memMods.toSeq) 
  }  

  // from Albert
  def createMemModule(m: DefMemory): Seq[DefModule] = {
    assert(m.dataType != UnknownType)
    val bbName = m.name + "_ext"
    val stmts = ArrayBuffer[Statement]()
    val wrapperioPorts = MemPortUtils.memToBundle(m).fields.map(f => Port(NoInfo, f.name, Input, f.tpe)) 
    val bbProto = m.copy(dataType = UIntType(IntWidth(bitWidth(m.dataType))))
    val bbioPorts = MemPortUtils.memToBundle(bbProto).fields.map(f => Port(NoInfo, f.name, Input, f.tpe)) 
    stmts += WDefInstance(m.info,bbName,bbName,UnknownType)
    val bbRef = createRef(bbName)
    stmts ++= (m.readers zip bbProto.readers).map{ 
      case (x,y) => adaptReader(createRef(x),m,createSubField(bbRef,y),bbProto)
    }.flatten 
    stmts ++= (m.writers zip bbProto.writers).map{ 
      case (x,y) => adaptWriter(createRef(x),m,createSubField(bbRef,y),bbProto)
    }.flatten 
    stmts ++= (m.readwriters zip bbProto.readwriters).map{ 
      case (x,y) => adaptReadWriter(createRef(x),m,createSubField(bbRef,y),bbProto)
    }.flatten  
    val wrapper = Module(m.info,m.name,wrapperioPorts,Block(stmts))   

    println(wrapper.body.serialize)

    val bb = ExtModule(m.info,bbName,bbioPorts) 
    Seq(bb,wrapper)
  }

  def adaptReader(aggPort: Expression, aggMem: DefMemory, groundPort: Expression, groundMem: DefMemory) = Seq(
    connectFields(groundPort,"addr",aggPort,"addr"),
    connectFields(groundPort,"en",aggPort,"en"),
    connectFields(groundPort,"clk",aggPort,"clk"),
    fromBits(
      WSubField(aggPort,"data",aggMem.dataType,UNKNOWNGENDER),
      WSubField(groundPort,"data",groundMem.dataType,UNKNOWNGENDER)
    )
  )

  def adaptWriter(aggPort: Expression, aggMem: DefMemory, groundPort: Expression, groundMem: DefMemory) = {
    val defaultSeq = Seq(
      connectFields(groundPort,"addr",aggPort,"addr"),
      connectFields(groundPort,"en",aggPort,"en"),
      connectFields(groundPort,"clk",aggPort,"clk"),
      Connect(
        NoInfo,
        WSubField(groundPort,"data",groundMem.dataType,UNKNOWNGENDER),
        toBits(WSubField(aggPort,"data",aggMem.dataType,UNKNOWNGENDER))
      )
    )
    if (containsInfo(aggMem.info,"maskGran"))
      defaultSeq :+ Connect(
        NoInfo,
        WSubField(groundPort,"mask",create_mask(groundMem.dataType),UNKNOWNGENDER),
        toBitMask(WSubField(aggPort,"mask",create_mask(aggMem.dataType),UNKNOWNGENDER),aggMem.dataType)
      )
    else defaultSeq
  }

  def adaptReadWriter(aggPort: Expression, aggMem: DefMemory, groundPort: Expression, groundMem: DefMemory) = {
    val defaultSeq = Seq(
      connectFields(groundPort,"addr",aggPort,"addr"),
      connectFields(groundPort,"en",aggPort,"en"),
      connectFields(groundPort,"clk",aggPort,"clk"),
      connectFields(groundPort,"wmode",aggPort,"wmode"),
      Connect(
        NoInfo,
        WSubField(groundPort,"wdata",groundMem.dataType,UNKNOWNGENDER),
        toBits(WSubField(aggPort,"wdata",aggMem.dataType,UNKNOWNGENDER))
      ),
      fromBits(
        WSubField(aggPort,"rdata",aggMem.dataType,UNKNOWNGENDER),
        WSubField(groundPort,"rdata",groundMem.dataType,UNKNOWNGENDER)
      )
    )
    if (containsInfo(aggMem.info,"maskGran"))
      defaultSeq :+ Connect(
        NoInfo,
        WSubField(groundPort,"wmask",create_mask(groundMem.dataType),UNKNOWNGENDER),
        toBitMask(WSubField(aggPort,"wmask",create_mask(aggMem.dataType),UNKNOWNGENDER),aggMem.dataType)
      )
    else defaultSeq
  }

}
