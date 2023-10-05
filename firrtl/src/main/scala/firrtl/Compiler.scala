// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.annotations._

/** Container of all annotations for a Firrtl compiler */
class AnnotationSeq private (underlying: Seq[Annotation]) {
  def toSeq: Seq[Annotation] = underlying
}
object AnnotationSeq {
  def apply(xs: Seq[Annotation]): AnnotationSeq = new AnnotationSeq(xs)
}
