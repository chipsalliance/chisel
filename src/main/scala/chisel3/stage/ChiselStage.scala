// See LICENSE for license details.

package chisel3.stage

import firrtl.AnnotationSeq
import firrtl.options.{Phase, PhaseManager, PreservesAll, Shell, Stage}
import firrtl.options.Viewer.view
import firrtl.stage.{FirrtlCli, FirrtlStage}

class ChiselStage extends Stage with PreservesAll[Phase] {
  val shell: Shell = new Shell("chisel") with ChiselCli with FirrtlCli

  private val phases: Seq[Phase] =
    new PhaseManager( Set( classOf[chisel3.stage.phases.Elaborate],
                           classOf[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
                           classOf[chisel3.stage.phases.Emitter],
                           classOf[chisel3.stage.phases.Convert] ) )
      .transformOrder
      .map(firrtl.options.phases.DeletedWrapper(_))

  def run(annotations: AnnotationSeq): AnnotationSeq = {
    val cOpts = view[ChiselOptions](annotations)

    /* @todo: Should this be wrapped in a try/catch? */
    (phases ++
       (if (cOpts.runFirrtlCompiler) { Some(new FirrtlStage) }
        else                         { Seq.empty         }))
      .foldLeft(annotations)( (a, f) => f.transform(a) )
  }

}
