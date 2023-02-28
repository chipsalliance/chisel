// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.internal.firrtl.Converter
import chisel3.stage.ChiselCircuitAnnotation
import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}
import firrtl.stage.FirrtlCircuitAnnotation

import scala.annotation.nowarn

/** This prepares a `ChiselCircuitAnnotation for compilation with FIRRTL. This does three things:
  *   - Uses `chisel3.internal.firrtl.Converter` to generate a FirrtlCircuitAnnotation`
  *   - Extracts all `firrtl.annotations.Annotation`s from the `chisel3.internal.firrtl.Circuit`
  *   - Generates any needed `RunFirrtlTransformAnnotation`s from extracted `firrtl.annotations.Annotation`s
  */
class Convert extends Phase {

  override def prerequisites = Seq(Dependency[Elaborate])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[MaybeInjectingPhase])
  override def invalidates(a: Phase) = false

  @nowarn("msg=Do not use annotations val of Circuit directly")
  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: ChiselCircuitAnnotation =>
      Some(a) ++
        /* Convert this Chisel Circuit to a FIRRTL Circuit */
        Some(FirrtlCircuitAnnotation(Converter.convert(a.circuit))) ++
        /* Convert all Chisel Annotations to FIRRTL Annotations */
        //TODO: clean up this code when firrtl is merged
        a.circuit.firrtlAnnotations
    case a => Some(a)
  }

}
