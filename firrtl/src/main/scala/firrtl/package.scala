// SPDX-License-Identifier: Apache-2.0

import firrtl.annotations.Annotation

package object firrtl {
  implicit def seqToAnnoSeq(xs: Seq[Annotation]): AnnotationSeq = AnnotationSeq(xs)
  implicit def annoSeqToSeq(as: AnnotationSeq):   Seq[Annotation] = as.toSeq
}
