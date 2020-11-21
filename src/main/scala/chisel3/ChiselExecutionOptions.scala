// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.stage.{NoRunFirrtlCompilerAnnotation, PrintFullStackTraceAnnotation}

import firrtl.{AnnotationSeq, ExecutionOptionsManager, ComposableOptions}

//TODO: provide support for running firrtl as separate process, could alternatively be controlled by external driver
//TODO: provide option for not saving chirrtl file, instead calling firrtl with in memory chirrtl
/**
  * Options that are specific to chisel.
  *
  * @param runFirrtlCompiler when true just run chisel, when false run chisel then compile its output with firrtl
  * @note this extends FirrtlExecutionOptions which extends CommonOptions providing easy access to down chain options
  */
case class ChiselExecutionOptions(
                                   runFirrtlCompiler: Boolean = true,
                                   printFullStackTrace: Boolean = false
                                   // var runFirrtlAsProcess: Boolean = false
                                 ) extends ComposableOptions {

  def toAnnotations: AnnotationSeq =
    (if (!runFirrtlCompiler) { Seq(NoRunFirrtlCompilerAnnotation) } else { Seq() }) ++
      (if (printFullStackTrace) { Some(PrintFullStackTraceAnnotation) } else { None })

}

trait HasChiselExecutionOptions {
  self: ExecutionOptionsManager =>

  var chiselOptions = ChiselExecutionOptions()

  parser.note("chisel3 options")

  parser.opt[Unit]("no-run-firrtl")
    .abbr("chnrf")
    .foreach { _ =>
      chiselOptions = chiselOptions.copy(runFirrtlCompiler = false)
    }
    .text("Stop after chisel emits chirrtl file")

  parser.opt[Unit]("full-stacktrace")
    .foreach { _ =>
      chiselOptions = chiselOptions.copy(printFullStackTrace = true)
    }
    .text("Do not trim stack trace")
}

