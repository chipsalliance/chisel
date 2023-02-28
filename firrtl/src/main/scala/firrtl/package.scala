// SPDX-License-Identifier: Apache-2.0

import firrtl.annotations.Annotation

package object firrtl {
  implicit def seqToAnnoSeq(xs: Seq[Annotation]) = AnnotationSeq(xs)
  implicit def annoSeqToSeq(as: AnnotationSeq): Seq[Annotation] = as.toSeq

  /* Options as annotations compatibility items */
  @deprecated("Use firrtl.options.TargetDirAnnotation", "FIRRTL 1.2")
  type TargetDirAnnotation = firrtl.options.TargetDirAnnotation

  @deprecated("Use firrtl.options.TargetDirAnnotation", "FIRRTL 1.2")
  val TargetDirAnnotation = firrtl.options.TargetDirAnnotation

  type WRef = ir.Reference
  type WSubField = ir.SubField
  type WSubIndex = ir.SubIndex
  type WSubAccess = ir.SubAccess
  type WDefInstance = ir.DefInstance
}
