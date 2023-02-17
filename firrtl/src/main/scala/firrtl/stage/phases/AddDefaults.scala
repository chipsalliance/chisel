// SPDX-License-Identifier: Apache-2.0

package firrtl.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, TargetDirAnnotation}
import firrtl.stage.TransformManager.TransformDependency
import firrtl.transforms.BlackBoxTargetDirAnno
import firrtl.stage.{FirrtlOptions, InfoModeAnnotation}

/** [[firrtl.options.Phase Phase]] that adds default [[FirrtlOption]] [[firrtl.annotations.Annotation Annotation]]s.
  * This is a part of the preprocessing done by [[FirrtlStage]].
  */
class AddDefaults extends Phase {

  override def prerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  /** Append any missing default annotations to an annotation sequence */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    var bb, im = true
    annotations.foreach {
      case _: BlackBoxTargetDirAnno => bb = false
      case _: InfoModeAnnotation    => im = false
      case _ =>
    }

    val default = new FirrtlOptions()
    val targetDir = annotations.collectFirst { case d: TargetDirAnnotation => d }
      .getOrElse(TargetDirAnnotation())
      .directory

    (if (bb) Seq(BlackBoxTargetDirAnno(targetDir)) else Seq()) ++
      (if (im) Seq(InfoModeAnnotation()) else Seq()) ++
      annotations
  }

}
