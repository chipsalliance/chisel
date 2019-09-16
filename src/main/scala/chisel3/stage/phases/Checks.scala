// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.stage.{ChiselOutputFileAnnotation, NoRunFirrtlCompilerAnnotation, PrintFullStackTraceAnnotation}

import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.options.{OptionsException, Phase, PreservesAll}

/** Sanity checks an [[firrtl.AnnotationSeq]] before running the main [[firrtl.options.Phase]]s of
  * [[chisel3.stage.ChiselStage]].
  */
class Checks extends Phase with PreservesAll[Phase] {

  override val dependents = Seq(classOf[Elaborate])

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val noF, st, outF = collection.mutable.ListBuffer[Annotation]()
    annotations.foreach {
      case a: NoRunFirrtlCompilerAnnotation.type => a +=: noF
      case a: PrintFullStackTraceAnnotation.type => a +=: st
      case a: ChiselOutputFileAnnotation         => a +=: outF
      case _ =>
    }

    if (noF.size > 1) {
      throw new OptionsException(
        s"""|At most one NoRunFirrtlCompilerAnnotation can be specified, but found '${noF.size}'. Did you duplicate:
            |    - option or annotation: -chnrf, --no-run-firrtl, NoRunFirrtlCompilerAnnotation
            |""".stripMargin)
    }

    if (st.size > 1) {
      throw new OptionsException(
        s"""|At most one PrintFullStackTraceAnnotation can be specified, but found '${noF.size}'. Did you duplicate:
            |    - option or annotation: --full-stacktrace, PrintFullStackTraceAnnotation
            |""".stripMargin)
    }

    if (outF.size > 1) {
      throw new OptionsException(
        s"""|At most one Chisel output file can be specified but found '${outF.size}'. Did you duplicate:
            |    - option or annotation: --chisel-output-file, ChiselOutputFileAnnotation
            |""".stripMargin)
    }

    annotations
  }

}
