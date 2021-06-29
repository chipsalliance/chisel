// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import firrtl.annotations.{Annotation, ReferenceTarget, SingleTargetAnnotation}

/** DecodeTableAnnotation used to store a decode result for a specific [[TruthTable]].
  * This is useful for saving large [[TruthTable]] during a elaboration time.
  *
  * @note user should manage the correctness of [[minimizedTable]].
  *
  * @param target output wire of a decoder.
  * @param truthTable input [[TruthTable]] encoded in a serialized [[TruthTable]].
  * @param minimizedTable minimized [[truthTable]] encoded in a serialized [[TruthTable]].
  */
case class DecodeTableAnnotation(
  target:         ReferenceTarget,
  truthTable:     String,
  minimizedTable: String)
    extends SingleTargetAnnotation[ReferenceTarget] {
  override def duplicate(n: ReferenceTarget): Annotation = this.copy(target = n)
}
