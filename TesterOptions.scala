// See LICENSE for license details.

package chisel3.iotesters

import java.io.File

import chisel3.HasChiselExecutionOptions
import firrtl.{HasFirrtlOptions, ComposableOptions, ExecutionOptionsManager}
import firrtl_interpreter.HasInterpreterOptions

import scala.collection.mutable


case class TesterOptions(
                          isGenVerilog:    Boolean = false,
                          isGenHarness:    Boolean = false,
                          isCompiling:     Boolean = false,
                          isRunTest:       Boolean = false,
                          isVerbose:       Boolean = false,
                          displayBase:     Int     = 10,
                          testerSeed:      Long    = System.currentTimeMillis,
                          testCmd:         mutable.ArrayBuffer[String]= mutable.ArrayBuffer[String](),
                          backendName:     String  = "firrtl",
                          logFileName:     String  = "",
                          waveform:        Option[File] = None) extends ComposableOptions

//Todo: Add options for isVerbose, displayBase, any othe missing things
trait HasTesterOptions {
  self: ExecutionOptionsManager =>

  var testerOptions = TesterOptions()

  parser.note("tester options")

  parser.opt[String]("backend-name")
    .abbr("tbn")
    .foreach { x => testerOptions = testerOptions.copy(backendName = x) }
    .text("run this as test command")

  parser.opt[Unit]("is-gen-verilog")
    .abbr("tigv")
    .foreach { _ => testerOptions = testerOptions.copy(isGenVerilog = true) }
    .text("has verilog already been generated")

  parser.opt[Unit]("is-gen-harness")
    .abbr("tigh")
    .foreach { _ => testerOptions = testerOptions.copy(isGenHarness = true) }
    .text("has harness already been generated")

  parser.opt[Unit]("is-compiling")
    .abbr("tic")
    .foreach { _ => testerOptions = testerOptions.copy(isCompiling = true) }
    .text("has harness already been generated")

  parser.opt[Seq[String]]("test-command")
    .abbr("ttc")
    .foreach { x => testerOptions = testerOptions.copy(testCmd = testerOptions.testCmd ++ x) }
    .text("run this as test command")

  parser.opt[String]("log-file-name")
    .abbr("tlfn")
    .foreach { x => testerOptions = testerOptions.copy(logFileName = x) }
    .text("write log file")

  parser.opt[File]("wave-form-file-name")
    .abbr("twffn")
    .foreach { x => testerOptions = testerOptions.copy(waveform = Some(x)) }
    .text("wave form file name")

  parser.opt[Long]("test-seed")
    .abbr("tts")
    .foreach { x => testerOptions = testerOptions.copy(testerSeed = x) }
    .text("provides a seed for random number generator")
}

class TesterOptionsManager
  extends ExecutionOptionsManager("chisel-testers")
    with HasTesterOptions
    with HasInterpreterOptions
    with HasChiselExecutionOptions
    with HasFirrtlOptions{
}

