// SPDX-License-Identifier: Apache-2.0

package firrtl.passes
package memlib

import firrtl.Utils.error
import firrtl._
import firrtl.annotations._
import firrtl.options.{CustomFileEmission, HasShellOptions, ShellOption}
import firrtl.passes.wiring._
import firrtl.stage.{Forms, RunFirrtlTransformAnnotation}

import java.io.{CharArrayWriter, PrintWriter}

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

case class ReplSeqMemAnnotation(inputFileName: String, outputConfig: String) extends NoTargetAnnotation

/** Generate conf file for a sequence of [[DefAnnotatedMemory]]
  * @note file already has its suffix adding by `--replSeqMem`
  */
case class MemLibOutConfigFileAnnotation(file: String, annotatedMemories: Seq[DefAnnotatedMemory])
    extends NoTargetAnnotation
    with CustomFileEmission {
  def baseFileName(annotations: AnnotationSeq) = file
  def suffix = None
  def getBytes = annotatedMemories.map { m =>
    require(bitWidth(m.dataType) <= Int.MaxValue)
    m.maskGran.foreach(x => require(x <= Int.MaxValue))
    MemConf(
      m.name,
      m.depth,
      bitWidth(m.dataType).toInt,
      m.readers.length,
      m.writers.length,
      m.readwriters.length,
      m.maskGran.map(_.toInt)
    ).toString
  }.mkString("\n").getBytes
}

private[memlib] case class AnnotatedMemoriesAnnotation(annotatedMemories: List[DefAnnotatedMemory])
    extends NoTargetAnnotation

object ReplSeqMemAnnotation {
  def parse(t: String): ReplSeqMemAnnotation = {
    val usage = """
[Optional] ReplSeqMem
  Pass to replace sequential memories with blackboxes + configuration file

Usage:
  --replSeqMem -c:<circuit>:-i:<filename>:-o:<filename>
  *** Note: sub-arguments to --replSeqMem should be delimited by : and not white space!

Required Arguments:
  -o<filename>         Specify the output configuration file
  -c<circuit>          Specify the target circuit

Optional Arguments:
  -i<filename>         Specify the input configuration file (for additional optimizations)
"""

    val passOptions = PassConfigUtil.getPassOptions(t, usage)
    val outputConfig = passOptions.getOrElse(
      OutputConfigFileName,
      error("No output config file provided for ReplSeqMem!" + usage)
    )
    val inputFileName = PassConfigUtil.getPassOptions(t).getOrElse(InputConfigFileName, "")
    ReplSeqMemAnnotation(inputFileName, outputConfig)
  }
}

@deprecated(
  "Migrate to a transform that does not take arguments. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class SimpleTransform(p: Pass, form: CircuitForm) extends Transform {
  def inputForm = form
  def outputForm = form
  def execute(state: CircuitState): CircuitState = CircuitState(p.run(state.circuit), state.form, state.annotations)
}

class SimpleMidTransform(p: Pass) extends SimpleTransform(p, MidForm)

// SimpleRun instead of PassBased because of the arguments to passSeq
class ReplSeqMem extends SeqTransform with HasShellOptions with DependencyAPIMigration {

  override def prerequisites = Forms.MidForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Forms.MidEmitters
  override def invalidates(a: Transform) = a match {
    case InferTypes | ResolveKinds | ResolveFlows | LowerTypes => true
    case _                                                     => false
  }

  val options = Seq(
    new ShellOption[String](
      longOption = "repl-seq-mem",
      toAnnotationSeq =
        (a: String) => Seq(passes.memlib.ReplSeqMemAnnotation.parse(a), RunFirrtlTransformAnnotation(new ReplSeqMem)),
      helpText = "Blackbox and emit a configuration file for each sequential memory",
      shortOption = Some("frsq"),
      helpValueName = Some("-c:<circuit>:-i:<file>:-o:<file>")
    )
  )

  val transforms: Seq[Transform] =
    Seq(
      new SimpleMidTransform(Legalize),
      new SimpleMidTransform(ToMemIR),
      new SimpleMidTransform(ResolveMaskGranularity),
      new SimpleMidTransform(RenameAnnotatedMemoryPorts),
      new CreateMemoryAnnotations,
      new ResolveMemoryReference,
      new ReplaceMemMacros,
      new WiringTransform,
      new DumpMemoryAnnotations
    )
}
