// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.aop.Aspect
import firrtl.AnnotationSeq
import firrtl.options.Phase

/** Run [[AspectPhase]] if a [[chisel3.aop.Aspect]] is present.
  */
class MaybeAspectPhase extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    if(annotations.collectFirst { case a: Aspect[_] => annotations }.isDefined) {
      new AspectPhase().transform(annotations)
    } else annotations
  }
}
