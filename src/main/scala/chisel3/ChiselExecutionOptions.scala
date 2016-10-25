// See LICENSE for license details.

package chisel3

import firrtl.{ExecutionOptionsManager, ComposableOptions}

//TODO: provide support for running firrtl as separate process, could alternatively be controlled by external driver
//TODO: provide option for not saving chirrtl file, instead calling firrtl with in memory chirrtl
/**
  * Options that are specific to chisel.
  *
  * @param runFirrtlCompiler when true just run chisel, when false run chisel then compile its output with firrtl
  * @note this extends FirrtlExecutionOptions which extends CommonOptions providing easy access to down chain options
  */
case class ChiselExecutionOptions(
                                   runFirrtlCompiler: Boolean = true
                                   // var runFirrtlAsProcess: Boolean = false
                                 ) extends ComposableOptions

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
}

