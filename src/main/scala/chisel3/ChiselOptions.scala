// See LICENSE for license details.

package chisel3

import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.HasScoptOptions
import scopt.OptionParser

/** Indicates that a subclass is an [[firrtl.annotation.Annotation]] with
  * an option consummable by [[HasChiselExecutionOptions]]
  *
  * This must be mixed into a subclass of [[annotaiton.Annotation]]
  */
sealed trait ChiselOption extends HasScoptOptions

/** Disables FIRRTL compiler execution
  *  - deasserts [[ChiselExecutionOptions.runFirrtlCompiler]]
  *  - equivalent to command line option `-chnrf/--no-run-firrtl`
  */
case object NoRunFirrtlAnnotation extends NoTargetAnnotation with ChiselOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[Unit]("no-run-firrtl")
    .abbr("chnrf")
    .action( (x, c) => c :+ NoRunFirrtlAnnotation )
    .text("Stop after chisel emits chirrtl file")
}

/** Disable saving CHIRRTL to an intermediate file
  *  - deasserts [[ChiselExecutionOptions.saveChirrtl]]
  *  - equivalent to command line option `--dont-save-chirrtl`
  */
case object DontSaveChirrtlAnnotation extends NoTargetAnnotation with ChiselOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[Unit]("dont-save-chirrtl")
    .action( (x, c) => c :+ DontSaveChirrtlAnnotation )
    .text("Do not save CHIRRTL output")
}

/** Disable saving CHIRRTL-time annotaitons to an intermediate file
  *  - deasserts [[ChiselExecutionOptions.saveAnnotations]]
  *  - equivalent to command line option `--dont-save-annotations`
  */
case object DontSaveAnnotationsAnnotation extends NoTargetAnnotation with ChiselOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[Unit]("dont-save-annotations")
    .action( (x, c) => c :+ DontSaveAnnotationsAnnotation )
    .text("Do not save Chisel Annotations")
}
