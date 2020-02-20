// See LICENSE for license details.

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.annotations.LegacyAnnotation
import firrtl.options.{Dependency, Phase, PreservesAll}

/** Convert any [[firrtl.annotations.LegacyAnnotation LegacyAnnotation]]s to non-legacy variants */
class ConvertLegacyAnnotations extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(Dependency[GetIncludes])

  override val dependents = Seq.empty

  def transform(annotations: AnnotationSeq): AnnotationSeq = LegacyAnnotation.convertLegacyAnnos(annotations)

}
