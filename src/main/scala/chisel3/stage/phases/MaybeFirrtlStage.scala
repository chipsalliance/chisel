// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.stage.NoRunFirrtlCompilerAnnotation

import firrtl.AnnotationSeq
import firrtl.options.Phase
import firrtl.stage.FirrtlStage

/** Run [[firrtl.stage.FirrtlStage]] if a [[chisel3.stage.NoRunFirrtlCompilerAnnotation]] is not present.
  */
class MaybeFirrtlStage extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
    .collectFirst { case NoRunFirrtlCompilerAnnotation => annotations }
    .getOrElse    { (new FirrtlStage).transform(annotations)          }

}
