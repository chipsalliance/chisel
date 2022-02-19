// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.stage.{
  CIRCTHandover,
  CIRCTOptions
}

import firrtl.AnnotationSeq
import firrtl.options.{
  Dependency,
  Phase
}
import firrtl.options.Viewer.view
import firrtl.stage.{
  Forms,
  RunFirrtlTransformAnnotation
}

class MaybeSFC extends Phase {

  override def prerequisites = Seq(Dependency[circt.stage.phases.AddDefaults])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[circt.stage.CIRCTStage])
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq) = {
    val sfcAnnotations: Option[AnnotationSeq] = view[CIRCTOptions](annotations).handover.get match {
      case CIRCTHandover.CHIRRTL      => None
      case CIRCTHandover.HighFIRRTL   => Some(Seq(RunFirrtlTransformAnnotation(new firrtl.HighFirrtlEmitter)))
      case CIRCTHandover.MiddleFIRRTL => Some(Seq(RunFirrtlTransformAnnotation(new firrtl.MiddleFirrtlEmitter)))
      case CIRCTHandover.LowFIRRTL    => Some(Seq(RunFirrtlTransformAnnotation(new firrtl.LowFirrtlEmitter)))
      case CIRCTHandover.LowOptimizedFIRRTL =>
        Some(
          (Forms.LowFormOptimized.toSet -- Forms.LowFormMinimumOptimized)
            .map(_.getObject())
            .map(RunFirrtlTransformAnnotation.apply).toSeq :+
            RunFirrtlTransformAnnotation(new firrtl.LowFirrtlEmitter)
        )
    }

    sfcAnnotations match {
      case Some(extra) =>
        logger.info(
          s"Running the Scala FIRRTL Compiler to CIRCT handover at ${view[CIRCTOptions](annotations).handover.get}"
        )
        /* Do not run custom transforms in the SFC.  This messes with scheduling. */
        val (toSFC, toMFC) = annotations.partition {
          case _: RunFirrtlTransformAnnotation => false
          case _ => true
        }
        (new firrtl.stage.phases.Compiler).transform(extra ++ toSFC) ++ toMFC
      case None => annotations
    }
  }

}
