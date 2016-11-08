// See license file for details

package firrtl.passes
package clocklist

import firrtl._
import firrtl.ir._
import Annotations._
import Utils.error
import java.io.{File, CharArrayWriter, PrintWriter, Writer}
import wiring.WiringUtils.{getChildrenMap, countInstances, ChildrenMap, getLineage}
import wiring.Lineage
import ClockListUtils._
import Utils._
import memlib.AnalysisUtils._
import memlib._
import Mappers._

case class ClockListAnnotation(target: ModuleName, outputConfig: String)
    extends Annotation with Loose with Unstable {
  def transform = classOf[ClockListTransform]
  def duplicate(n: Named) = n match {
    case m: ModuleName => this.copy(target = m, outputConfig = outputConfig)
    case _ => error("Clocklist can only annotate a module.")
  }
}

object ClockListAnnotation {
  def apply(t: String) = {
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
    new ClockListAnnotation(target, outputConfig)
  }
}

class ClockListTransform extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm
  def passSeq(top: String, writer: Writer): Seq[Pass] =
    Seq(new ClockList(top, writer))
  def execute(state: CircuitState): CircuitState = getMyAnnotations(state) match {
    case Seq(ClockListAnnotation(ModuleName(top, CircuitName(state.circuit.main)), out)) => 
      val outputFile = new PrintWriter(out)
      val newC = (new ClockList(top, outputFile)).run(state.circuit)
      outputFile.close()
      CircuitState(newC, state.form)
    case Nil => CircuitState(state.circuit, state.form)
    case seq => error(s"Found illegal clock list annotation(s): $seq")
  }
}
