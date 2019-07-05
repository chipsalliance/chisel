// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.aop.Aspect
import firrtl.AnnotationSeq
import firrtl.options.Phase

/** Run [[firrtl.stage.FirrtlStage]] if a [[chisel3.stage.NoRunFirrtlCompilerAnnotation]] is not present.
  */
class MaybeAspectStage extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    if(annotations.collectFirst { case a: Aspect[_] => annotations }.isDefined) {
      new chisel3.stage.AspectStage().transform(annotations)
    } else annotations
  }
}
