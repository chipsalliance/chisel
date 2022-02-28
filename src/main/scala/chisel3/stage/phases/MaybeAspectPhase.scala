// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.aop.Aspect
import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}

/** Run [[AspectPhase]] if a [[chisel3.aop.Aspect]] is present.
  */
class MaybeAspectPhase extends Phase {

  override def prerequisites = Seq(Dependency[Elaborate])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    if (annotations.collectFirst { case a: Aspect[_] => annotations }.isDefined) {
      new AspectPhase().transform(annotations)
    } else annotations
  }
}
