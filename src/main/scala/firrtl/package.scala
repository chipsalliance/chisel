// See LICENSE for license details.

import firrtl.AnnotationSeq
import firrtl.annotations.Annotation

package object firrtl {
  implicit def seqToAnnoSeq(xs: Seq[Annotation]) = AnnotationSeq(xs)
  implicit def annoSeqToSeq(as: AnnotationSeq): Seq[Annotation] = as.underlying

  /* Options as annotations compatibility items */
  @deprecated("Use firrtl.stage.TargetDirAnnotation", "3.2")
  type TargetDirAnnotation = firrtl.stage.TargetDirAnnotation

  @deprecated("Use firrtl.stage.TargetDirAnnotation", "3.2")
  val TargetDirAnnotation = firrtl.stage.TargetDirAnnotation
}
