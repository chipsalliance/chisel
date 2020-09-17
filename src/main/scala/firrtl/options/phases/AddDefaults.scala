// SPDX-License-Identifier: Apache-2.0

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, TargetDirAnnotation}

/** Add default annotations for a [[Stage]]
  *
  * This currently only adds a [[TargetDirAnnotation]]. This isn't necessary for a [[StageOptionsView]], but downstream
  * tools may expect a [[TargetDirAnnotation]] to exist.
  */
class AddDefaults extends Phase {

  override def prerequisites = Seq(Dependency[GetIncludes])

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val td = annotations.collectFirst { case a: TargetDirAnnotation => a }.isEmpty

    (if (td) Seq(TargetDirAnnotation()) else Seq()) ++
      annotations
  }

}
