// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import firrtl.annotations._
import AnalysisUtils._
import Utils.error
import java.io.{File, CharArrayWriter, PrintWriter}
import wiring._

sealed trait PassOption
case object InputConfigFileName extends PassOption
case object OutputConfigFileName extends PassOption
case object PassCircuitName extends PassOption
case object PassModuleName extends PassOption

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
        case "-m" :: value :: tail =>
          nextPassOption(map + (PassModuleName -> value), tail)
        case option :: tail =>
          error("Unknown option " + option + usage)
      }
    }
    nextPassOption(Map[PassOption, String](), passArgList)
  }
}

class ConfWriter(filename: String) {
  val outputBuffer = new CharArrayWriter
  def append(m: DefAnnotatedMemory) = {
    // legacy
    val maskGran = m.maskGran
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

object ReplSeqMemAnnotation {
  def apply(t: String): Annotation = {
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
    val inputFileName = PassConfigUtil.getPassOptions(t).getOrElse(InputConfigFileName, "")
    val passCircuit = passOptions.getOrElse(
      PassCircuitName, 
      error("No circuit name specified for ReplSeqMem!" + usage)
    )
    val target = CircuitName(passCircuit)
    Annotation(target, classOf[ReplSeqMem], s"$inputFileName $outputConfig")
  }

  def apply(target: CircuitName, inputFileName: String, outputConfig: String): Annotation =
    Annotation(target, classOf[ReplSeqMem], s"$inputFileName $outputConfig")

  private val matcher = "([^ ]*) ([^ ]+)".r
  def unapply(a: Annotation): Option[(CircuitName, String, String)] = a match {
    case Annotation(CircuitName(c), t, matcher(inputFileName, outputConfig)) if t == classOf[ReplSeqMem] =>
      Some((CircuitName(c), inputFileName, outputConfig))
    case _ => None
  }
}

class SimpleTransform(p: Pass, form: CircuitForm) extends Transform {
  def inputForm = form
  def outputForm = form
  def execute(state: CircuitState): CircuitState = CircuitState(p.run(state.circuit), state.form, state.annotations)
}

class SimpleMidTransform(p: Pass) extends SimpleTransform(p, MidForm)

// SimpleRun instead of PassBased because of the arguments to passSeq
class ReplSeqMem extends Transform {
  def inputForm = MidForm
  def outputForm = MidForm
  def transforms(inConfigFile: Option[YamlFileReader], outConfigFile: ConfWriter): Seq[Transform] =
    Seq(new SimpleMidTransform(Legalize),
        new SimpleMidTransform(ToMemIR),
        new SimpleMidTransform(ResolveMaskGranularity),
        new SimpleMidTransform(RenameAnnotatedMemoryPorts),
        new ResolveMemoryReference,
        new CreateMemoryAnnotations(inConfigFile),
        new ReplaceMemMacros(outConfigFile),
        new WiringTransform,
        new SimpleMidTransform(RemoveEmpty),
        new SimpleMidTransform(CheckInitialization),
        new SimpleMidTransform(InferTypes),
        Uniquify,
        new SimpleMidTransform(ResolveKinds),
        new SimpleMidTransform(ResolveGenders))

  def execute(state: CircuitState): CircuitState = getMyAnnotations(state) match {
    case Nil => state // Do nothing if there are no annotations
    case p => (p.collectFirst { case a if (a.target == CircuitName(state.circuit.main)) => a }) match {
      case Some(ReplSeqMemAnnotation(target, inputFileName, outputConfig)) =>
        val inConfigFile = {
          if (inputFileName.isEmpty) None 
          else if (new File(inputFileName).exists) Some(new YamlFileReader(inputFileName))
          else error("Input configuration file does not exist!")
        }
        val outConfigFile = new ConfWriter(outputConfig)
        transforms(inConfigFile, outConfigFile).foldLeft(state) { (in, xform) => xform.runTransform(in) }
      case _ => error("Unexpected transform annotation")
    }
  }
}
