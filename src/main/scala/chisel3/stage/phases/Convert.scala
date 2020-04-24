// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.experimental.RunFirrtlTransform
import chisel3.internal.firrtl.Converter
import chisel3.stage.ChiselCircuitAnnotation
import firrtl.{AnnotationSeq, Transform}
import firrtl.options.{Dependency, Phase, PreservesAll}
import firrtl.stage.{FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}

/** This prepares a [[ChiselCircuitAnnotation]] for compilation with FIRRTL. This does three things:
  *   - Uses [[chisel3.internal.firrtl.Converter]] to generate a [[FirrtlCircuitAnnotation]]
  *   - Extracts all [[firrtl.annotations.Annotation]]s from the [[chisel3.internal.firrtl.Circuit]]
  *   - Generates any needed [[RunFirrtlTransformAnnotation]]s from extracted [[firrtl.annotations.Annotation]]s
  */
class Convert extends Phase {

  override val prerequisites = Seq(Dependency[Elaborate])

  override def invalidates(phase: Phase): Boolean = phase match {
    case _: Elaborate => true
    case _ => false
  }

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: ChiselCircuitAnnotation =>
      /* Convert this Chisel Circuit to a FIRRTL Circuit */
      Some(FirrtlCircuitAnnotation(Converter.convert(a.circuit))) ++
      /* Convert all Chisel Annotations to FIRRTL Annotations */
      a
        .circuit
        .annotations
        .map(_.toFirrtl) ++
      a
        .circuit
        .annotations
        .collect {
          case anno: RunFirrtlTransform => anno.transformClass
        }
        .distinct
        .map { c: Class[_ <: Transform] => RunFirrtlTransformAnnotation(c.newInstance()) }
    case a => Some(a)
  }

}
