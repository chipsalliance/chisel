// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.experimental.RunFirrtlTransform
import chisel3.internal.firrtl.Converter
import chisel3.stage.ChiselCircuitAnnotation

import firrtl.{AnnotationSeq, Transform}
import firrtl.annotations.DeletedAnnotation
import firrtl.options.Phase
import firrtl.stage.{FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}

object Convert extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.flatMap {
    case a: ChiselCircuitAnnotation =>
      /* Convert this Chisel Circuit to a FIRRTL Circuit */
      Seq( DeletedAnnotation(name, a), FirrtlCircuitAnnotation(Converter.convert(a.circuit)) ) ++
        /* Convert all Chisel Annotations to FIRRTL Annotations */
        (a.circuit
           .annotations
           .map(_.toFirrtl)) ++
        /* Add requested FIRRTL Transforms for any Chisel Annotations which mixed in RunFirrtlTransform */
        (a.circuit
           .annotations
           .collect { case b: RunFirrtlTransform => b.transformClass }
           .distinct
           .filterNot(_ == classOf[firrtl.Transform])
           .map { c: Class[_ <: Transform] => RunFirrtlTransformAnnotation(c.newInstance()) })
    case a => Seq(a)
  }

}
