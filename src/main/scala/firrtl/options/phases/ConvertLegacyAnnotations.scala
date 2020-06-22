// See LICENSE for license details.

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.annotations.LegacyAnnotation
import firrtl.options.{Dependency, Phase}

/** Convert any [[firrtl.annotations.LegacyAnnotation LegacyAnnotation]]s to non-legacy variants */
class ConvertLegacyAnnotations extends Phase {

  override def prerequisites = Seq(Dependency[GetIncludes])

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = LegacyAnnotation.convertLegacyAnnos(annotations)

}
