// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import Annotations._
import AnalysisUtils._
import Utils.error
import java.io.{File, CharArrayWriter, PrintWriter}

sealed trait PassOption
case object InputConfigFileName extends PassOption
case object OutputConfigFileName extends PassOption
case object PassCircuitName extends PassOption

object PassConfigUtil {
  type PassOptionMap = Map[PassOption, String]
 
  def getPassOptions(t: String, usage: String = "") = {
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
  val outputBuffer = new CharArrayWriter
  def append(m: DefMemory) = {
    // legacy
    val maskGran = getInfo(m.info, "maskGran")
    val readers = List.fill(m.readers.length)("read")
    val writers = List.fill(m.writers.length)(if (maskGran.isEmpty) "write" else "mwrite")
    val readwriters = List.fill(m.readwriters.length)(if (maskGran.isEmpty) "rw" else "mrw")
    val ports = (writers ++ readers ++ readwriters) mkString ","
    val maskGranConf = maskGran match { case None => "" case Some(p) => s"mask_gran $p" }
    val width = bitWidth(m.dataType)
    val conf = s"name ${m.name} depth ${m.depth} width $width ports $ports $maskGranConf \n"
    outputBuffer.append(conf)
  }
  def serialize() = {
    val outputFile = new PrintWriter(filename)
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
  def duplicate(n: Named) = this copy (t = (t replace (s"-c:$passCircuit", s"-c:${n.name}")))
}

class ReplSeqMem(transID: TransID) extends Transform with SimpleRun {
  def passSeq(inConfigFile: Option[YamlFileReader], outConfigFile: ConfWriter) =
    Seq(Legalize,
        AnnotateMemMacros,
        UpdateDuplicateMemMacros,
        new AnnotateValidMemConfigs(inConfigFile),
        new ReplaceMemMacros(outConfigFile),
        RemoveEmpty,
        CheckInitialization,
        InferTypes,
        ResolveKinds,         // Must be run for the transform to work!
        ResolveGenders)

  def execute(c: Circuit, map: AnnotationMap) = map get transID match {
    case Some(p) => p get CircuitName(c.main) match {
      case Some(ReplSeqMemAnnotation(t, _)) =>
        val inputFileName = PassConfigUtil.getPassOptions(t).getOrElse(InputConfigFileName, "")
        val inConfigFile = {
          if (inputFileName.isEmpty) None 
          else if (new File(inputFileName).exists) Some(new YamlFileReader(inputFileName))
          else error("Input configuration file does not exist!")
        }
        val outConfigFile = new ConfWriter(PassConfigUtil.getPassOptions(t)(OutputConfigFileName))
        run(c, passSeq(inConfigFile, outConfigFile))
      case _ => error("Unexpected transform annotation")
    }
    case _ => TransformResult(c)
  }
}
