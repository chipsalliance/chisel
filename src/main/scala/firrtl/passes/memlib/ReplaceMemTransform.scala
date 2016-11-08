// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import Annotations._
import AnalysisUtils._
import Utils.error
import java.io.{File, CharArrayWriter, PrintWriter}
import wiring._

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

case class ReplSeqMemAnnotation(t: String) extends Annotation with Loose with Unstable {

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
  def duplicate(n: Named) = this copy (t = t.replace(s"-c:$passCircuit", s"-c:${n.name}"))
  def transform = classOf[ReplSeqMem]
}

class SimpleTransform(p: Pass, form: CircuitForm) extends Transform {
  def inputForm = form
  def outputForm = form
  def execute(state: CircuitState): CircuitState = CircuitState(p.run(state.circuit), state.form)
}

class SimpleMidTransform(p: Pass) extends SimpleTransform(p, MidForm)

// SimpleRun instead of PassBased because of the arguments to passSeq
class ReplSeqMem extends Transform with SimpleRun {
  def inputForm = MidForm
  def outputForm = MidForm
  def passSeq(inConfigFile: Option[YamlFileReader], outConfigFile: ConfWriter): Seq[Transform] =
    Seq(new SimpleMidTransform(Legalize),
        new SimpleMidTransform(ToMemIR),
        new SimpleMidTransform(ResolveMaskGranularity),
        new SimpleMidTransform(RenameAnnotatedMemoryPorts),
        new SimpleMidTransform(ResolveMemoryReference),
        new CreateMemoryAnnotations(inConfigFile),
        new ReplaceMemMacros(outConfigFile),
        new WiringTransform,
        new SimpleMidTransform(RemoveEmpty),
        new SimpleMidTransform(CheckInitialization),
        new SimpleMidTransform(InferTypes),
        new SimpleMidTransform(Uniquify),
        new SimpleMidTransform(ResolveKinds),
        new SimpleMidTransform(ResolveGenders))
  def run(state: CircuitState, xForms: Seq[Transform]): CircuitState = {
    (xForms.foldLeft(state) { case (curState: CircuitState, xForm: Transform) =>
      val res = xForm.execute(curState)
      val newAnnotations = res.annotations match {
        case None => curState.annotations
        case Some(ann) => 
          Some(AnnotationMap(ann.annotations ++ curState.annotations.get.annotations))
      }
      CircuitState(res.circuit, res.form, newAnnotations)
    }).copy(annotations = None)
  }

  def execute(state: CircuitState): CircuitState =
    getMyAnnotations(state) match {
      case Nil => state.copy(annotations = None) // Do nothing if there are no annotations
      case p => (p.collectFirst { case a if (a.target == CircuitName(state.circuit.main)) => a }) match {
        case Some(ReplSeqMemAnnotation(t)) =>
          val inputFileName = PassConfigUtil.getPassOptions(t).getOrElse(InputConfigFileName, "")
          val inConfigFile = {
            if (inputFileName.isEmpty) None 
            else if (new File(inputFileName).exists) Some(new YamlFileReader(inputFileName))
            else error("Input configuration file does not exist!")
          }
          val outConfigFile = new ConfWriter(PassConfigUtil.getPassOptions(t)(OutputConfigFileName))
          run(state, passSeq(inConfigFile, outConfigFile))
        case _ => error("Unexpected transform annotation")
      }
    }
}
