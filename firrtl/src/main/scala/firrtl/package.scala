// SPDX-License-Identifier: Apache-2.0

import firrtl.annotations.Annotation

package object firrtl {
  @deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
  implicit def seqToAnnoSeq(xs: Seq[Annotation]): AnnotationSeq = AnnotationSeq(xs)
  @deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
  implicit def annoSeqToSeq(as: AnnotationSeq): Seq[Annotation] = as.toSeq
}
