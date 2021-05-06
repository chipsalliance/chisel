// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.decode

import firrtl.annotations.{Annotation, ReferenceTarget, SingleTargetAnnotation}

case class DecodeTableAnnotation(
  target:         ReferenceTarget,
  truthTable:     TruthTable,
  minimizedTable: TruthTable)
    extends SingleTargetAnnotation[ReferenceTarget] {
  override def duplicate(n: ReferenceTarget): Annotation = this.copy(target = n)
}
