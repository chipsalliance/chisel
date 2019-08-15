// See LICENSE for license details.

package chisel3.stage

import firrtl.AnnotationSeq
import firrtl.options.{Phase, PhaseManager, PreservesAll, Shell, Stage}
import firrtl.options.phases.DeletedWrapper
import firrtl.stage.FirrtlCli

class ChiselStage extends Stage with PreservesAll[Phase] {
  val shell: Shell = new Shell("chisel") with ChiselCli with FirrtlCli

  private val targets =
    Seq( classOf[chisel3.stage.phases.Checks],
         classOf[chisel3.stage.phases.Elaborate],
         classOf[chisel3.stage.phases.AddImplicitOutputFile],
         classOf[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
         classOf[chisel3.stage.phases.MaybeAspectPhase],
         classOf[chisel3.stage.phases.Emitter],
         classOf[chisel3.stage.phases.Convert],
         classOf[chisel3.stage.phases.MaybeFirrtlStage] )

  def run(annotations: AnnotationSeq): AnnotationSeq =
    /* @todo: Should this be wrapped in a try/catch? */
    new PhaseManager(targets) { override val wrappers = Seq( (a: Phase) => DeletedWrapper(a) ) }
      .transformOrder
      .map(firrtl.options.phases.DeletedWrapper(_))
      .foldLeft(annotations)( (a, f) => f.transform(a) )

}
