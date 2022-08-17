// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlFileAnnotation}

import chisel3.stage.CircuitSerializationAnnotation

private[stage] class AddFIRRTLInputFile extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[circt.stage.CIRCTStage])
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: CircuitSerializationAnnotation => Some(FirrtlFileAnnotation(a.filename(annotations).toString))
    case a: FirrtlCircuitAnnotation        => None
    case a => Some(a)
  }

}
