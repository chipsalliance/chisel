// See LICENSE for license details.

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.annotations.LegacyAnnotation
import firrtl.options.Phase

/** Convert any [[firrtl.annotations.LegacyAnnotation LegacyAnnotation]]s to non-legacy variants */
class ConvertLegacyAnnotations extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = LegacyAnnotation.convertLegacyAnnos(annotations)

}
