// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.aop.Aspect
import firrtl.AnnotationSeq
import firrtl.options.{Phase, PreservesAll}

/** Run [[AspectPhase]] if a [[chisel3.aop.Aspect]] is present.
  */
class MaybeAspectPhase extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(classOf[Elaborate])

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    if(annotations.collectFirst { case a: Aspect[_] => annotations }.isDefined) {
      new AspectPhase().transform(annotations)
    } else annotations
  }
}
