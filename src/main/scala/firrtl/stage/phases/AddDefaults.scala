// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.{AnnotationSeq, VerilogEmitter}
import firrtl.options.{Dependency, Phase, TargetDirAnnotation}
import firrtl.stage.TransformManager.TransformDependency
import firrtl.transforms.BlackBoxTargetDirAnno
import firrtl.stage.{FirrtlOptions, InfoModeAnnotation, RunFirrtlTransformAnnotation}

/** [[firrtl.options.Phase Phase]] that adds default [[FirrtlOption]] [[firrtl.annotations.Annotation Annotation]]s.
  * This is a part of the preprocessing done by [[FirrtlStage]].
  */
class AddDefaults extends Phase {

  override def prerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  val DefaultEmitterTarget: TransformDependency = Dependency[VerilogEmitter]

  /** Append any missing default annotations to an annotation sequence */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    var bb, em, im = true
    annotations.foreach {
      case _: BlackBoxTargetDirAnno => bb = false
      case _: InfoModeAnnotation    => im = false
      case RunFirrtlTransformAnnotation(_: firrtl.Emitter) => em = false
      case _ =>
    }

    val default = new FirrtlOptions()
    val targetDir = annotations.collectFirst { case d: TargetDirAnnotation => d }
      .getOrElse(TargetDirAnnotation())
      .directory

    (if (bb) Seq(BlackBoxTargetDirAnno(targetDir)) else Seq()) ++
      // if there is no compiler or emitter specified, add the default emitter
      (if (em) Seq(RunFirrtlTransformAnnotation(DefaultEmitterTarget)) else Seq()) ++
      (if (im) Seq(InfoModeAnnotation()) else Seq()) ++
      annotations
  }

}
