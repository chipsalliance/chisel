// See LICENSE for license details.

package firrtl.passes

import com.typesafe.scalalogging.LazyLogging
import firrtl._
import firrtl.ir._
import Annotations._
import java.io.Writer
import AnalysisUtils._

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
          error("Unknown option " + option + usage)
      }
    }
    nextPassOption(Map[PassOption, String](), passArgList)
  }

}

class ConfWriter(filename: String) {
  val outputBuffer = new java.io.CharArrayWriter
  def append(m: DefMemory) = {
    // legacy
    val maskGran = getInfo(m.info, "maskGran")
    val writers = m.writers map (x => if (maskGran == None) "write" else "mwrite")
    val readers = List.fill(m.readers.length)("read")
    val readwriters = m.readwriters map (x => if (maskGran == None) "rw" else "mrw")
    val ports = (writers ++ readers ++ readwriters).mkString(",")
    val maskGranConf = if (maskGran == None) "" else s"mask_gran ${maskGran.get}"
    val width = bitWidth(m.dataType)
    val conf = s"name ${m.name} depth ${m.depth} width ${width} ports ${ports} ${maskGranConf} \n"
    outputBuffer.append(conf)
  }
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
  --replSeqMem -c:<circuit>:-i:<filename>:-o:<filename>
  *** Note: sub-arguments to --replSeqMem should be delimited by : and not white space!

Required Arguments:
  -o<filename>         Specify the output configuration file
  -c<compiler>         Specify the target circuit

Optional Arguments:
  -i<filename>         Specify the input configuration file (for additional optimizations)
"""    

  val passOptions = PassConfigUtil.getPassOptions(t, usage)
  val outputConfig = passOptions.getOrElse(
    OutputConfigFileName, 
    error("No output config file provided for ReplSeqMem!" + usage)
  )
  val passCircuit = passOptions.getOrElse(
    PassCircuitName, 
    error("No circuit name specified for ReplSeqMem!" + usage)
  )
  val target = CircuitName(passCircuit)
  def duplicate(n: Named) = this.copy(t=t.replace("-c:"+passCircuit, "-c:"+n.name))
  
}

class ReplSeqMem(transID: TransID) extends Transform with LazyLogging {
  def execute(circuit:Circuit, map: AnnotationMap) = 
    map get transID match {
      case Some(p) => p get CircuitName(circuit.main) match {
        case Some(ReplSeqMemAnnotation(t, _)) => {

          val inputFileName = PassConfigUtil.getPassOptions(t).getOrElse(InputConfigFileName, "")
          val inConfigFile = {
            if (inputFileName.isEmpty) None 
            else if (new java.io.File(inputFileName).exists) Some(new YamlFileReader(inputFileName))
            else error("Input configuration file does not exist!")
          }

          val outConfigFile = new ConfWriter(PassConfigUtil.getPassOptions(t).get(OutputConfigFileName).get)
          TransformResult(
            (
              Seq(
                Legalize,
                AnnotateMemMacros,
                UpdateDuplicateMemMacros,
                new AnnotateValidMemConfigs(inConfigFile),
                new ReplaceMemMacros(outConfigFile),
                RemoveEmpty,
                CheckInitialization,
                ResolveKinds,                                       // Must be run for the transform to work!
                InferTypes,
                ResolveGenders
              ) foldLeft circuit
            ) { 
              (c, pass) =>
                val x = Utils.time(pass.name)(pass run c)
                logger debug x.serialize
                x
            } , 
            None, 
            Some(map)
          )
        }  
        case _ => error("Unexpected transform annotation")
      }
      case _ => TransformResult(circuit, None, Some(map))
    }
}