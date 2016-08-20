package firrtl.passes

import com.typesafe.scalalogging.LazyLogging
import scala.collection.mutable.{ArrayBuffer,HashMap}

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.Utils._
import Annotations._
import firrtl.PrimOps._
import firrtl.WrappedExpression._

import java.io.Writer

import scala.util.matching.Regex

sealed trait PassOption
case object InputConfigFileName extends PassOption
case object OutputConfigFileName extends PassOption
case object PassCircuitName extends PassOption

object PassConfigUtil {

  def getPassOptions(t: String, usage: String = "") = {
    
    type PassOptionMap = Map[PassOption, String] 

    // can't use space to delimit sub arguments (otherwise, Driver.scala will throw error)
    val passArgList = t.split(":").toList
    
    def nextPassOption(map: PassOptionMap, list: List[String]): PassOptionMap = {
      list match {
        case Nil => map
        case "-i" :: value :: tail =>
          nextPassOption(map + (InputConfigFileName -> value), tail)
        case "-o" :: value :: tail =>
          nextPassOption(map + (OutputConfigFileName -> value), tail)
        case "-c" :: value :: tail =>
          nextPassOption(map + (PassCircuitName -> value), tail)
        case option :: tail =>
          throw new Exception("Unknown option " + option + usage)
      }
    }
    nextPassOption(Map[PassOption, String](), passArgList)
  }

}

class OutputWriter(filename: String) {
  val outputBuffer = new java.io.CharArrayWriter
  def append(s: String) = outputBuffer.append(s)
  def serialize = {
    val outputFile = new java.io.PrintWriter(filename)
    outputFile.write(outputBuffer.toString)
    outputFile.close()
  }
}

case class ReplSeqMemAnnotation(t: String, tID: TransID)
    extends Annotation with Loose with Unstable {

  val usage = """
[Optional] ReplSeqMem
  Pass to replace sequential memories with blackboxes + configuration file

Usage: 
  --replSeqMem -c:<circuit>:-i<filename>:-o<filename>
  *** Note: sub-arguments to --replSeqMem should be delimited by : and not white space!

Required Arguments:
  -o<filename>         Specify the output configuration file
  -c<compiler>         Specify the target circuit

Optional Arguments:
  -i<filename>         Specify the input configuration file
"""    

  val passOptions = PassConfigUtil.getPassOptions(t,usage)
  val outputConfig = passOptions.getOrElse(OutputConfigFileName, throw new Exception("No output config file provided for ReplSeqMem!" + usage))
  val passCircuit = passOptions.getOrElse(PassCircuitName, throw new Exception("No circuit name specified for ReplSeqMem!" + usage))
  val target = CircuitName(passCircuit)
  def duplicate(n: Named) = this.copy(t=t.replace("-c:"+passCircuit,"-c:"+n.name))
  
}

class ReplSeqMemPass(out: OutputWriter) extends Pass {

  def name = "Replace Sequential Memories with Blackboxes + Configuration File"

  trait WritePortChar {
    def name: String
    def useMask: Boolean
    def maskGran: Option[BigInt]
    require( (useMask && (maskGran != None)) || (!useMask), "Must specify a mask granularity if write mask is desired" )
  }

  case class PortForWrite(
    name: String,
    useMask: Boolean = false,
    maskGran: Option[BigInt] = None
  ) extends WritePortChar 

  case class PortForReadWrite(
    name: String,
    useMask: Boolean = false,
    maskGran: Option[BigInt] = None
  ) extends WritePortChar 

  case class PortForRead(
    name: String
  )

  // vendor agnostic configuration
  case class SMem(
    m: DefMemory,
    // names of read ports
    readPorts: Seq[PortForRead],
    // write ports
    writePorts: Seq[PortForWrite],
    // read/write ports
    readWritePorts: Seq[PortForReadWrite]
  ){
    require ( 
      if (readWritePorts.isEmpty) writePorts.nonEmpty && readPorts.nonEmpty else writePorts.isEmpty && readPorts.isEmpty,
      "Need at least one set of read, write ports if no RW port is specified. A RW port must be standalone"
    )  
    require (readWritePorts.length < 2, "Cannot have more than 1 readwrite port")
    def name = m.name
    def dataType = m.dataType
    def depth = m.depth
    def writeLatency = m.writeLatency
    def readLatency = m.readLatency
    def numReaders = readPorts.length
    def numWriters = writePorts.length
    def numRWriters = readWritePorts.length  
    def rPortMap = readPorts.zipWithIndex map { case (p,i) => p -> s"R$i" }
    def wPortMap = writePorts.zipWithIndex map { case (p,i) => p.name -> s"W$i" }
    def rwPortMap = readWritePorts.zipWithIndex map { case (p,i) => p.name -> s"RW$i" }
    def width = bitWidth(dataType)
    def serialize = {
      // for backwards compatibility with old conf format
      val writers = writePorts map (x => if (x.useMask) "mwrite" else "write")
      val readers = List.fill(numReaders)("read")
      val readwriters = readWritePorts map (x => if(x.useMask) "mrw" else "rw")
      val ports = (writers ++ readers ++ readwriters).mkString(",")
      // old conf file only supported 1 mask_gran
      val maskGran = (writePorts ++ readWritePorts) map (_.maskGran.getOrElse(0))
      val maskGranConf = if (maskGran.head == 0) "" else s"mask_gran ${maskGran.head}"
      s"name ${name} depth ${depth} width ${width} ports ${ports} ${maskGranConf} \n"
    }
    def eq(m: SMem) = {
      // TODO: Condition on read under write
      val wpIndivEq = writePorts zip m.writePorts map {case(a,b) => a.maskGran == b.maskGran} 
      val wpEq = wpIndivEq.foldLeft(true)(_ && _)
      val rwpIndivEq = readWritePorts zip m.readWritePorts map {case(a,b) => a.maskGran == b.maskGran} 
      val rwpEq = rwpIndivEq.foldLeft(true)(_ && _)
      (dataType == m.dataType) && 
      (depth == m.depth) && 
      (writeLatency == m.writeLatency) && 
      (readLatency == m.readLatency) && 
      (numReaders == m.numReaders) && 
      (wpEq && rwpEq)
    }
    def getInterfacePorts = MemPortUtils.memToBundle(m).fields.map(f => Port(NoInfo, f.name, Input, f.tpe))
  }

  def analyzeMemsInModule(m: Module): Seq[SMem] = {

    val connects = HashMap[String, Expression]()
    val mems = ArrayBuffer[SMem]()

    // swiped from InferRW 
    def findConnects(s: Statement): Unit = s match {
      case s: Connect  =>
        connects(s.loc.serialize) = s.expr
      case s: PartialConnect =>
        connects(s.loc.serialize) = s.expr
      case s: DefNode =>
        connects(s.name) = s.value
      case s: Block =>
        s.stmts foreach findConnects
      case _ =>
    }

    def findConnectOriginFromExp(e: Expression): Seq[Expression] = e match {
      // matches how wmode, wmask, write_en are assigned (from Chirrtl) 
      // in case no ConstProp is performed before this pass
      case Mux(cond, tv, fv, _) if we(tv) == we(one) && we(fv) == we(zero) =>
        cond +: findConnectOrigin(cond.serialize)
      // visit connected nodes to references
      case _: WRef | _: SubField | _: SubIndex | _: SubAccess =>
        e +: findConnectOrigin(e.serialize) 
      // backward searches until a PrimOp or Literal appears -->
      // Literal: you've reached origin
      // PrimOp: you're not simply doing propagation anymore
      // NOTE: not a catch-all!!! 
      case _ => List(e)  
    }

    // only capable of searching for origin in the same module
    def findConnectOrigin(node: String): Seq[Expression] = {
      if (connects contains node) findConnectOriginFromExp(connects(node))
      else Nil
    }

    // returns None if wen = wmask bits or wmask bits all = 1; otherwise returns # of mask bits
    def getMaskBits(wen: String, wmask: String): Option[Int] = {
      val wenOrigin = findConnectOrigin(wen)
      // find all mask bits
      val wmaskOrigin = connects.keys.toSeq filter (_.startsWith(wmask)) map findConnectOrigin
      val bitEq = wmaskOrigin map (wenOrigin intersect _) map (_.length > 0) 
      // when all wmask bits are equal to wmode, wmask is redundant
      val eq = bitEq.foldLeft(true)(_ && _)
      val wmaskBitOne = wmaskOrigin map(_ contains one)
      // if all wmask bits = 1, then wmask is redundant
      val wmaskOne = wmaskBitOne.foldLeft(true)(_ && _)
      if (eq || wmaskOne) None else Some(wmaskOrigin.length)
    }

    def findMemInsts(s: Statement): Unit = s match {
      // only find smems
      case m: DefMemory if m.readLatency > 0 =>
        val dataBits = bitWidth(m.dataType)
        val rwPorts = m.readwriters map (w => {
          val maskBits = getMaskBits(s"${m.name}.$w.wmode",s"${m.name}.$w.wmask")
          if (maskBits == None) PortForReadWrite(name = w)
          else PortForReadWrite(name = w, useMask = true, maskGran = Some(dataBits/maskBits.get))
        })
        val wPorts = m.writers map (w => {
          val maskBits = getMaskBits(s"${m.name}.$w.en",s"${m.name}.$w.mask")
          if (maskBits == None) PortForWrite(name = w)
          else PortForWrite(name = w, useMask = true, maskGran = Some(dataBits/maskBits.get))  
        })
        val smemInfo = SMem(
          m = m,
          readPorts = m.readers map(r => PortForRead(name = r)),
          writePorts = wPorts,
          readWritePorts = rwPorts
        )
        mems += smemInfo
      case b: Block => b.stmts foreach findMemInsts
      case _ => 
    }
    findConnects(m.body)
    findMemInsts(m.body)
    mems.toSeq
  }

  def run(c: Circuit) = {
    lazy val moduleNamespace = Namespace(c)

    val uniqueMems = ArrayBuffer[SMem]()
    val mems = ArrayBuffer[SMem]()
    def analyzeMemsInCircuit(c: Circuit) = {
      c.modules foreach { 
        case m: Module => mems ++= analyzeMemsInModule(m)
        case m: ExtModule =>
      }
      mems map {m =>
        val memProto = uniqueMems.find(_.eq(m))
        if (memProto == None) {
          uniqueMems += m
          m.name -> m
        }
        else m.name -> memProto.get.copy(m=m.m)
      }
    }
    val memMap = analyzeMemsInCircuit(c)
    val newMods = mems map (m => ExtModule(m.m.info,m.name,m.getInterfacePorts)) 

    def replaceMemInstsInCircuit(c: Circuit) = {
      def replaceMemInstsInModule(m: Module) = {
        def findMemInsts(s: Statement): Statement = s match {
          case m: DefMemory if m.readLatency > 0 =>  WDefInstance(m.info, m.name, m.name, UnknownType) 
          case b: Block => Block(b.stmts map findMemInsts)
          case s => s
        }
        m.copy(body = findMemInsts(m.body))
      }
      c.modules map {
        case m: Module => replaceMemInstsInModule(m)
        case m: ExtModule => m
      } 
    } 

    uniqueMems foreach { m =>
      moduleNamespace.newName(m.name)
      moduleNamespace.newName(m.name + "_ext")
      out.append(m.serialize)
    }
    out.serialize
    c.copy(modules = replaceMemInstsInCircuit(c) ++ newMods)
  }

}

class ReplSeqMem(transID: TransID) extends Transform with LazyLogging {
  def execute(circuit:Circuit, map: AnnotationMap) = 
    map get transID match {
      case Some(p) => p get CircuitName(circuit.main) match {
        case Some(ReplSeqMemAnnotation(t, _)) => {
          val outConfigFile = PassConfigUtil.getPassOptions(t).get(OutputConfigFileName).get
          TransformResult(
            (
              Seq(
                Legalize,
                new ReplSeqMemPass(new OutputWriter(outConfigFile)),
                RemoveEmpty,
                CheckInitialization,
                ResolveKinds,                                       // Must be run for the transform to work!
                InferTypes,
                ResolveGenders
              ) foldLeft circuit
            ){ 
              (c, pass) =>
                val x = Utils.time(pass.name)(pass run c)
                logger debug x.serialize
                x
            }, 
            None, 
            Some(map)
          )
        }  
        case _ => TransformResult(circuit, None, Some(map))
      }
      case _ => TransformResult(circuit, None, Some(map))
    }
}

// Eliminate extra modules
// Tag modules
// connect internals