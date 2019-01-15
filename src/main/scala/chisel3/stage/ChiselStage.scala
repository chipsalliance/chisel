// See LICENSE for license details.

package chisel3.stage

import firrtl.AnnotationSeq
import firrtl.options.{Phase, Shell, Stage}
import firrtl.options.Viewer.view
import firrtl.stage.{FirrtlCli, FirrtlStage}

import logger.{Logger, LoggerCli}

object ChiselStage extends Stage {
  val shell: Shell = new Shell("chisel") with LoggerCli with ChiselCli with FirrtlCli

  private val phases: Seq[Phase] = Seq(
    chisel3.stage.phases.Checks,
    chisel3.stage.phases.Elaborate,
    chisel3.stage.phases.AddImplicitOutputFile,
    chisel3.stage.phases.AddImplicitOutputAnnotationFile,
    chisel3.stage.phases.Emitter,
    chisel3.stage.phases.Convert
  )

  def run(annotations: AnnotationSeq): AnnotationSeq = Logger.makeScope(annotations) {
    val cOpts = view[ChiselOptions](annotations)

    /* @todo: Should this be wrapped in a try/catch? */
    (phases ++
       (if (cOpts.runFirrtlCompiler) { Some(FirrtlStage) }
        else                         { Seq.empty         }))
      .foldLeft(annotations)( (a, f) => f.transform(a) )
  }

}
