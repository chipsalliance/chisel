// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.Implicits.BooleanImplicits
import circt.stage.CIRCTHandover

import firrtl.AnnotationSeq
import firrtl.options.Phase

class AddDefaults extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {

    var handover = false
    annotations.foreach {
      case _: CIRCTHandover => handover = true
      case _ =>
    }

    annotations ++
      (!handover).option(CIRCTHandover(CIRCTHandover.LowOptimizedFIRRTL))
  }

}
