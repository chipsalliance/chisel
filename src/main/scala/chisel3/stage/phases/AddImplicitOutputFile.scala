// See LICENSE for license details.

package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, PreservesAll}

import chisel3.stage.{ChiselCircuitAnnotation, ChiselOutputFileAnnotation}

/** Add a output file for a Chisel circuit, derived from the top module in the circuit, if no
  * [[ChiselOutputFileAnnotation]] already exists.
  */
class AddImplicitOutputFile extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(Dependency[Elaborate])

  def transform(annotations: AnnotationSeq): AnnotationSeq =
    annotations.collectFirst{ case _: ChiselOutputFileAnnotation  => annotations }.getOrElse{

      val x: Option[AnnotationSeq] = annotations
        .collectFirst{ case a: ChiselCircuitAnnotation =>
          ChiselOutputFileAnnotation(a.circuit.name) +: annotations }

      x.getOrElse(annotations)
    }

}
