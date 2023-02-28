// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, PhaseManager, Shell, Stage, StageMain}
import firrtl.options.phases.DeletedWrapper
import firrtl.stage.phases.CatchExceptions

class FirrtlPhase
    extends PhaseManager(
      targets = Seq(
        Dependency[firrtl.stage.phases.Compiler],
        Dependency[firrtl.stage.phases.ConvertCompilerAnnotations]
      )
    ) {

  override def invalidates(a: Phase) = false

  override val wrappers = Seq(CatchExceptions(_: Phase), DeletedWrapper(_: Phase))

}

class FirrtlStage extends Stage {

  lazy val phase = new FirrtlPhase

  override def prerequisites = phase.prerequisites

  override def optionalPrerequisites = phase.optionalPrerequisites

  override def optionalPrerequisiteOf = phase.optionalPrerequisiteOf

  override def invalidates(a: Phase): Boolean = phase.invalidates(a)

  val shell: Shell = new Shell("firrtl") with FirrtlCli

  override protected def run(annotations: AnnotationSeq): AnnotationSeq = phase.transform(annotations)

}

object FirrtlMain extends StageMain(new FirrtlStage)
