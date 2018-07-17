// See LICENSE for license details.

package chisel3

import firrtl.{AnnotationSeq, FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}
import firrtl.annotations.{NoTargetAnnotation, SingleTargetAnnotation, CircuitName}
import firrtl.options.HasScoptOptions
import chisel3.experimental.{RawModule, RunFirrtlTransform}
import chisel3.internal.firrtl.{Converter, Circuit}
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
    .unbounded()
    .text("Stop after chisel emits chirrtl file")
}

/** Disable saving CHIRRTL to an intermediate file
  *  - deasserts [[ChiselExecutionOptions.saveChirrtl]]
  *  - equivalent to command line option `--dont-save-chirrtl`
  */
case object DontSaveChirrtlAnnotation extends NoTargetAnnotation with ChiselOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[Unit]("dont-save-chirrtl")
    .action( (x, c) => c :+ DontSaveChirrtlAnnotation )
    .unbounded()
    .text("Do not save CHIRRTL output")
}

/** Disable saving CHIRRTL-time annotaitons to an intermediate file
  *  - deasserts [[ChiselExecutionOptions.saveAnnotations]]
  *  - equivalent to command line option `--dont-save-annotations`
  */
case object DontSaveAnnotationsAnnotation extends NoTargetAnnotation with ChiselOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[Unit]("dont-save-annotations")
    .action( (x, c) => c :+ DontSaveAnnotationsAnnotation )
    .unbounded()
    .text("Do not save Chisel Annotations")
}

/** Holds a Chisel circuit
  *
  * @param target a circuit name
  * @param circuit the Chisel Circuit
  * @note this is not JSON serializable and will be unpacked into a
  * [[firrtl.FirrtlCircuitAnnotation]] and
  * [[firrtl.RunFirrtlTransformAnnotation]] by Chisel's Driver
  */
case class ChiselCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with ChiselOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[String]("dut")
    .action{ (x, c) => ChiselCircuitAnnotation(() => Class.forName(x).newInstance().asInstanceOf[RawModule]) +: c }
    .validate( x => try {
                Class.forName(x)
                p.success
              } catch {
                case e: ClassNotFoundException =>
                  throw new ChiselException(s"Class $x not found (did you misspell it?", null) })
    .unbounded()
    .text("Chisel design under test to load (must be of type <: RawModule)")
}

object ChiselCircuitAnnotation {
  /** Elaborate a Chisel circuit from a lambda and create a circuit annotation
    *
    * @param dut a function that creates a Chisel circuit
    */
  def apply(dut: () => RawModule): ChiselCircuitAnnotation = ChiselCircuitAnnotation(Driver.elaborate(dut))

  private [chisel3] def apply(): ChiselCircuitAnnotation = ChiselCircuitAnnotation(Circuit("null", Seq.empty))
}
