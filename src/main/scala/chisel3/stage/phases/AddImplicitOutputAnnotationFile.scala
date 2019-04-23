// See LICENSE for license details.

package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{OutputAnnotationFileAnnotation, Phase, PreservesAll, StageOptions}
import firrtl.options.Viewer.view

import chisel3.stage.ChiselCircuitAnnotation

/** Adds an [[firrtl.options.OutputAnnotationFileAnnotation OutputAnnotationFileAnnotation]] if one does not exist. This
  * replicates old behavior where an output annotation file was always written.
  */
class AddImplicitOutputAnnotationFile extends Phase with PreservesAll[Phase] {

  override val prerequisites: Set[Class[Phase]] = Set(classOf[Checks], classOf[Elaborate])

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
    .collectFirst{ case a: OutputAnnotationFileAnnotation => annotations }
    .getOrElse{

      val x: Option[AnnotationSeq] = annotations
        .collectFirst{ case a: ChiselCircuitAnnotation =>
          OutputAnnotationFileAnnotation(a.circuit.name) +: annotations }

      x.getOrElse(annotations)
    }

}
