// See LICENSE for license details.

package firrtl.stage

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, PhaseManager, PreservesAll, Shell, Stage, StageMain}
import firrtl.options.phases.DeletedWrapper
import firrtl.stage.phases.CatchExceptions

class FirrtlPhase
    extends PhaseManager(targets=Seq(Dependency[firrtl.stage.phases.Compiler], Dependency[firrtl.stage.phases.WriteEmitted]))
    with PreservesAll[Phase] {

  override val wrappers = Seq(CatchExceptions(_: Phase), DeletedWrapper(_: Phase))

}

class FirrtlStage extends Stage {

  lazy val phase = new FirrtlPhase

  override lazy val prerequisites = phase.prerequisites

  override lazy val dependents = phase.dependents

  override def invalidates(a: Phase): Boolean = phase.invalidates(a)

  val shell: Shell = new Shell("firrtl") with FirrtlCli

  def run(annotations: AnnotationSeq): AnnotationSeq = phase.transform(annotations)

}

object FirrtlMain extends StageMain(new FirrtlStage)
