// See LICENSE for license details.

package chisel3

import firrtl.{
  AnnotationSeq,
  FirrtlViewer }
import firrtl.options.{
  ExecutionOptionsManager,
  OptionsView,
  Viewer }
import firrtl.annotations.{
  Annotation,
  NoTargetAnnotation }

/** Indicates that a subclass is an [[firrtl.annotation.Annotation]] with
  * an option consummable by [[HasChiselExecutionOptions]]
  *
  * This must be mixed into a subclass of [[annotaiton.Annotation]]
  */
sealed trait ChiselOption { self: Annotation => }

/** Disables FIRRTL compiler execution
  *  - deasserts [[ChiselExecutionOptions.runFirrtlCompiler]]
  *  - equivalent to command line option `-chnrf/--no-run-firrtl`
  */
case object NoRunFirrtlAnnotation extends NoTargetAnnotation with ChiselOption

/** Disable saving CHIRRTL to an intermediate file
  *  - deasserts [[ChiselExecutionOptions.saveChirrtl]]
  *  - equivalent to command line option `--dont-save-chirrtl`
  */
case object DontSaveChirrtlAnnotation extends NoTargetAnnotation with ChiselOption

/** Disable saving CHIRRTL-time annotaitons to an intermediate file
  *  - deasserts [[ChiselExecutionOptions.saveAnnotations]]
  *  - equivalent to command line option `--dont-save-annotations`
  */
case object DontSaveAnnotationsAnnotation extends NoTargetAnnotation with ChiselOption

// TODO: provide support for running firrtl as separate process, could
//       alternatively be controlled by external driver
/** Options that control the execution of the Chisel compiler
  *
  * @param runFirrtlCompiler run the FIRRTL compiler if true
  * @param saveChirrtl save CHIRRTL output to a file if true
  * @param saveAnnotations save CHIRRTL-time annotations to a file if true
  * @note this extends FirrtlExecutionOptions which extends CommonOptions providing easy access to down chain options
  */
case class ChiselExecutionOptions (
  runFirrtlCompiler: Boolean = true,
  saveChirrtl: Boolean       = true,
  saveAnnotations: Boolean   = true
)

object ChiselViewer {
  implicit object ChiselOptionsView extends OptionsView[ChiselExecutionOptions] {
    def view(implicit options: AnnotationSeq): Option[ChiselExecutionOptions] = Some(
      options
        .collect{ case opt: ChiselOption => opt }
        .foldLeft(ChiselExecutionOptions())( (c, x) =>
          x match {
            case NoRunFirrtlAnnotation         => c.copy(runFirrtlCompiler = false)
            case DontSaveChirrtlAnnotation     => c.copy(saveChirrtl = false)
            case DontSaveAnnotationsAnnotation => c.copy(saveAnnotations = false)
          } ) )
  }
}

trait HasChiselExecutionOptions { this: ExecutionOptionsManager =>
  import firrtl.options.Viewer._
  import chisel3.ChiselViewer._

  /** A [[ChiselExecutionOptions]] object generated from processing all
    * Chisel command line options
    */
  @deprecated("Use view[ChiselExecutionOptions]", "3.2.0")
  lazy val chiselOptions: ChiselExecutionOptions = view[ChiselExecutionOptions].getOrElse{
    throw new Exception("Unable to parse Chisel command line options/annotations") }

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
