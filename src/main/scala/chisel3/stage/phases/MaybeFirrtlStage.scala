// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import chisel3.stage.NoRunFirrtlCompilerAnnotation

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}
import firrtl.stage.FirrtlStage

/** Run [[firrtl.stage.FirrtlStage]] if a [[chisel3.stage.NoRunFirrtlCompilerAnnotation]] is not present.
  */
class MaybeFirrtlStage extends Phase {

  override def prerequisites = Seq(Dependency[Convert])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = annotations.collectFirst {
    case NoRunFirrtlCompilerAnnotation => annotations
  }.getOrElse { (new FirrtlStage).transform(annotations) }

}
