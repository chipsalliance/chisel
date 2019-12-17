// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Phase, PreservesAll, TargetDirAnnotation}
import firrtl.transforms.BlackBoxTargetDirAnno
import firrtl.stage.{CompilerAnnotation, InfoModeAnnotation, FirrtlOptions}

/** [[firrtl.options.Phase Phase]] that adds default [[FirrtlOption]] [[firrtl.annotations.Annotation Annotation]]s.
  * This is a part of the preprocessing done by [[FirrtlStage]].
  */
class AddDefaults extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq.empty

  override val dependents = Seq.empty

  /** Append any missing default annotations to an annotation sequence */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    var bb, c, im = true
    annotations.foreach {
      case _: BlackBoxTargetDirAnno => bb = false
      case _: CompilerAnnotation => c  = false
      case _: InfoModeAnnotation => im = false
      case a =>
    }

    val default = new FirrtlOptions()
    val targetDir = annotations
      .collectFirst { case d: TargetDirAnnotation => d }
      .getOrElse(TargetDirAnnotation()).directory

    (if (bb) Seq(BlackBoxTargetDirAnno(targetDir)) else Seq() ) ++
      (if (c) Seq(CompilerAnnotation(default.compiler)) else Seq() ) ++
      (if (im) Seq(InfoModeAnnotation()) else Seq() ) ++
      annotations
  }

}
