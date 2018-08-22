// See license file for details

package firrtl.passes
package clocklist

import firrtl._
import firrtl.ir._
import annotations._
import Utils.error
import java.io.{File, CharArrayWriter, PrintWriter, Writer}
import wiring.Lineage
import ClockListUtils._
import Utils._
import memlib.AnalysisUtils._
import memlib._
import Mappers._
import firrtl.options.RegisteredTransform
import scopt.OptionParser
import firrtl.stage.RunFirrtlTransformAnnotation

case class ClockListAnnotation(target: ModuleName, outputConfig: String) extends
    SingleTargetAnnotation[ModuleName] {
  def duplicate(n: ModuleName) = ClockListAnnotation(n, outputConfig)
}

object ClockListAnnotation {
  def parse(t: String): ClockListAnnotation = {
    val usage = """
[Optional] ClockList
  List which signal drives each clock of every descendent of specified module

Usage:
  --list-clocks -c:<circuit>:-m:<module>:-o:<filename>
  *** Note: sub-arguments to --list-clocks should be delimited by : and not white space!
"""

    //Parse pass options
    val passOptions = PassConfigUtil.getPassOptions(t, usage)
    val outputConfig = passOptions.getOrElse(
      OutputConfigFileName,
      error("No output config file provided for ClockList!" + usage)
    )
    val passCircuit = passOptions.getOrElse(
      PassCircuitName,
      error("No circuit name specified for ClockList!" + usage)
    )
    val passModule = passOptions.getOrElse(
      PassModuleName,
      error("No module name specified for ClockList!" + usage)
    )
    passOptions.get(InputConfigFileName) match {
      case Some(x) => error("Unneeded input config file name!" + usage)
      case None =>
    }
    val target = ModuleName(passModule, CircuitName(passCircuit))
    ClockListAnnotation(target, outputConfig)
  }
}

class ClockListTransform extends Transform with RegisteredTransform {
  def inputForm = LowForm
  def outputForm = LowForm

  def addOptions(parser: OptionParser[AnnotationSeq]): Unit = parser
    .opt[String]("list-clocks")
    .abbr("clks")
    .valueName ("-c:<circuit>:-m:<module>:-o:<filename>")
    .action( (x, c) => c ++ Seq(passes.clocklist.ClockListAnnotation.parse(x),
                                RunFirrtlTransformAnnotation(new ClockListTransform)) )
    .maxOccurs(1)
    .text("List which signal drives each clock of every descendent of specified module")

  def passSeq(top: String, writer: Writer): Seq[Pass] =
    Seq(new ClockList(top, writer))
  def execute(state: CircuitState): CircuitState = {
    val annos = state.annotations.collect { case a: ClockListAnnotation => a }
    annos match {
      case Seq(ClockListAnnotation(ModuleName(top, CircuitName(state.circuit.main)), out)) =>
        val outputFile = new PrintWriter(out)
        val newC = (new ClockList(top, outputFile)).run(state.circuit)
        outputFile.close()
        CircuitState(newC, state.form, state.annotations)
      case Nil => state
      case seq => error(s"Found illegal clock list annotation(s): $seq")
    }
  }
}
