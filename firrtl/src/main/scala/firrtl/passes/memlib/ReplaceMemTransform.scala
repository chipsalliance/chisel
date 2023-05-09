// SPDX-License-Identifier: Apache-2.0

package firrtl.passes
package memlib

import firrtl.Utils.error
import firrtl._
import firrtl.annotations._
import firrtl.options.{HasShellOptions, ShellOption}

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

object ReplSeqMemAnnotation extends HasShellOptions {
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

  val options = Seq(
    new ShellOption[String](
      longOption = "repl-seq-mem",
      toAnnotationSeq = (a: String) => Seq(passes.memlib.ReplSeqMemAnnotation.parse(a)),
      helpText = "Blackbox and emit a configuration file for each sequential memory",
      shortOption = Some("frsq"),
      helpValueName = Some("-c:<circuit>:-i:<file>:-o:<file>")
    )
  )
}
