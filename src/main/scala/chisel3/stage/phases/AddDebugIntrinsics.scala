// SPDX-License-Identifier: Apache-2.0
package chisel3.stage.phases

import chisel3.debug.{DebugIntrinsics, EmitDebugIntrinsicsAnnotation}
import chisel3.stage.ChiselCircuitAnnotation

import firrtl.options.{Dependency, Phase}
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}

/** No-op unless [[chisel3.debug.EmitDebugIntrinsicsAnnotation]] is present.
  *
  * @note This API is experimental and subject to change.
  */
class AddDebugIntrinsics extends Phase {
  override def prerequisites = Seq(Dependency[Elaborate])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[Convert], Dependency[AddDedupGroupAnnotations])
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq =
    if (!annotations.contains(EmitDebugIntrinsicsAnnotation)) annotations
    else
      // Drop the annotation after consumption so a second pass is a no-op
      // (the circuit is already mutated; re-running would duplicate intrinsics).
      annotations.flatMap {
        case EmitDebugIntrinsicsAnnotation => Nil
        case a: ChiselCircuitAnnotation =>
          DebugIntrinsics.generate(a.elaboratedCircuit._circuit)
          Seq(a)
        case a => Seq(a)
      }
}
