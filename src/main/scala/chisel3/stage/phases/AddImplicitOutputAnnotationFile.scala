// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.stage.ChiselCircuitAnnotation
import firrtl.AnnotationSeq
import firrtl.options.{OutputAnnotationFileAnnotation, Phase, PreservesAll}

/** Adds an [[firrtl.options.OutputAnnotationFileAnnotation]] if one does not exist. This replicates old behavior where
  * an output annotation file was always written.
  */
class AddImplicitOutputAnnotationFile extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(classOf[Elaborate])

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
    .collectFirst{ case _: OutputAnnotationFileAnnotation => annotations }
    .getOrElse{

      val x: Option[AnnotationSeq] = annotations
        .collectFirst{ case a: ChiselCircuitAnnotation =>
          OutputAnnotationFileAnnotation(a.circuit.name) +: annotations }

      x.getOrElse(annotations)
    }

}
