// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.annotations._

/** Container of all annotations for a Firrtl compiler */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
class AnnotationSeq private (underlying: Seq[Annotation]) {
  def toSeq: Seq[Annotation] = underlying
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object AnnotationSeq {
  def apply(xs: Seq[Annotation]): AnnotationSeq = new AnnotationSeq(xs)
}
