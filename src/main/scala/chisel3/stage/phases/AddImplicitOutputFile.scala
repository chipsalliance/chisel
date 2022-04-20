// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}

import chisel3.stage.{ChiselCircuitAnnotation, ChiselOutputFileAnnotation}

/** Add a output file for a Chisel circuit, derived from the top module in the circuit, if no
  * [[ChiselOutputFileAnnotation]] already exists.
  */
class AddImplicitOutputFile extends Phase {

  override def prerequisites = Seq(Dependency[Elaborate])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq =
    annotations.collectFirst { case _: ChiselOutputFileAnnotation => annotations }.getOrElse {

      val x: Option[AnnotationSeq] = annotations.collectFirst {
        case a: ChiselCircuitAnnotation =>
          ChiselOutputFileAnnotation(a.circuit.name) +: annotations
      }

      x.getOrElse(annotations)
    }

}
