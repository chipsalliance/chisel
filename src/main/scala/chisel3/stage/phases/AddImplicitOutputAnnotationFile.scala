// See LICENSE for license details.

package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{OutputAnnotationFileAnnotation, Phase, StageOptions}
import firrtl.options.Viewer.view

import chisel3.stage.ChiselCircuitAnnotation

/** Adds an [[firrtl.options.OutputAnnotationFileAnnotation OutputAnnotationFileAnnotation]] if one does not exist. This
  * replicates old behavior where an output annotation file was always written.
  */
object AddImplicitOutputAnnotationFile extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
    .collectFirst{ case a: OutputAnnotationFileAnnotation => annotations }
    .getOrElse{

      val x: Option[AnnotationSeq] = annotations
        .collectFirst{ case a: ChiselCircuitAnnotation =>
          OutputAnnotationFileAnnotation(a.circuit.name) +: annotations }

      x.getOrElse(annotations)
    }

}
