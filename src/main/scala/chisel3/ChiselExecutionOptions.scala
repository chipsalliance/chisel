// See LICENSE for license details.

package chisel3

import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.options.{ExecutionOptionsManager, OptionsView, Viewer}
import chisel3.internal.firrtl.Circuit

class ChiselOptionsException(msg: String) extends ChiselException(msg, null)

// TODO: provide support for running firrtl as separate process, could
//       alternatively be controlled by external driver
/** Options that control the execution of the Chisel compiler
  *
  * @param runFirrtlCompiler run the FIRRTL compiler if true
  * @param saveChirrtl save CHIRRTL output to a file if true
  * @param saveAnnotations save CHIRRTL-time annotations to a file if true
  * @param chiselCircuit a Chisel circuit
  */
case class ChiselExecutionOptions (
  runFirrtlCompiler: Boolean     = true,
  saveChirrtl: Boolean           = true,
  saveAnnotations: Boolean       = true,
  chiselCircuit: Option[Circuit] = None
)

object ChiselViewer {
  implicit object ChiselOptionsView extends OptionsView[ChiselExecutionOptions] {
    def checkAnnotations(annos: AnnotationSeq): AnnotationSeq = {
      val c = collection.mutable.ListBuffer[Annotation]()
      annos.foreach{
        case a: ChiselCircuitAnnotation => c += a
        case _                          =>
      }
      if (c.isEmpty) {
        throw new ChiselOptionsException("No Chisel circuit specified via ChiselCircuitAnnotation or --dut") }
      if (c.size > 1) {
        throw new ChiselOptionsException(
          "Only one Chisel circuit can be specified but found multiple ChiselCircuitAnnotation or --dut arguments") }
      annos
    }

    def view(options: AnnotationSeq): Option[ChiselExecutionOptions] = {
      val annotationTransforms = Seq(checkAnnotations(_))

      val preprocessedAnnotations = annotationTransforms.foldLeft(options)( (old, tx) => tx(old) )

      val (chiselAnnos, nonChiselAnnos) = preprocessedAnnotations.partition {
        case opt: ChiselOption => true
        case _                 => false }

      val x = chiselAnnos
        .foldLeft(ChiselExecutionOptions())(
          (c, x) => x match {
            case NoRunFirrtlAnnotation           => c.copy(runFirrtlCompiler = false)
            case DontSaveChirrtlAnnotation       => c.copy(saveChirrtl       = false)
            case DontSaveAnnotationsAnnotation   => c.copy(saveAnnotations   = false)
            case a: ChiselCircuitAnnotation      => c.copy(chiselCircuit     = Some(a.circuit)) })
      Some(x)
    }
  }
}

trait HasChiselExecutionOptions { this: ExecutionOptionsManager =>
  parser.note("Chisel Options")

  // [todo] This could be handled with reflection via knownDirectSubclasses
  Seq( NoRunFirrtlAnnotation,
       DontSaveChirrtlAnnotation,
       DontSaveAnnotationsAnnotation,
       ChiselCircuitAnnotation() )
    .map(_.addOptions(parser))
}
