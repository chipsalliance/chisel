// See LICENSE for license details.

package chisel3

import firrtl.ExecutionOptionsManager
import firrtl.annotations.NoTargetAnnotation

sealed trait ChiselOption
case object NoRunFirrtlAnnotation extends NoTargetAnnotation with ChiselOption
case object DontSaveChirrtlAnnotation extends NoTargetAnnotation with ChiselOption
case object DontSaveAnnotationsAnnotation extends NoTargetAnnotation with ChiselOption

//TODO: provide support for running firrtl as separate process, could alternatively be controlled by external driver
/**
  * Options that are specific to chisel.
  *
  * @param runFirrtlCompiler when true just run chisel, when false run chisel then compile its output with firrtl
  * @note this extends FirrtlExecutionOptions which extends CommonOptions providing easy access to down chain options
  */
case class ChiselExecutionOptions (
  runFirrtlCompiler: Boolean = true,
  saveChirrtl: Boolean = true,
  saveAnnotations: Boolean = true
)

trait HasChiselExecutionOptions {
  self: ExecutionOptionsManager =>

  lazy val chiselOptions: ChiselExecutionOptions = options
    .collect{ case opt: ChiselOption => opt }
    .foldLeft(ChiselExecutionOptions())( (old, x) =>
      x match {
        case NoRunFirrtlAnnotation => old.copy(runFirrtlCompiler = false)
        case DontSaveChirrtlAnnotation => old.copy(saveChirrtl = false)
        case DontSaveAnnotationsAnnotation => old.copy(saveAnnotations = false)
      } )

  parser.note("Chisel Options")

  parser.opt[Unit]("no-run-firrtl")
    .abbr("chnrf")
    .action( (x, c) => c :+ NoRunFirrtlAnnotation )
    .text("Stop after chisel emits chirrtl file")

  parser.opt[Unit]("dont-save-chirrtl")
    .action( (x, c) => c :+ DontSaveChirrtlAnnotation )
    .text("Do not save CHIRRTL output")

  parser.opt[Unit]("dont-save-annotations")
    .action( (x, c) => c :+ DontSaveAnnotationsAnnotation )
    .text("Do not save Chisel Annotations")
}
