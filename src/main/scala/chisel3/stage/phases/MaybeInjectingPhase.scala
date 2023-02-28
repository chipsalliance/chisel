// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.aop.injecting.{InjectStatement, InjectingPhase}
import firrtl.AnnotationSeq
import firrtl.options.Phase

/** Run `InjectingPhase` if a `InjectStatement` is present.
  */
class MaybeInjectingPhase extends Phase {
  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false
  def transform(annotations:  AnnotationSeq): AnnotationSeq = annotations.collectFirst {
    case _: InjectStatement => new InjectingPhase().transform(annotations)
  }.getOrElse(annotations)
}
