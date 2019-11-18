// See LICENSE for license details.

package firrtl

import firrtl.annotations.DeletedAnnotation
import firrtl.options.{OptionsView, Viewer}
import firrtl.stage.phases.WriteEmitted

/** The [[stage]] package provides an implementation of the FIRRTL compiler using the [[firrtl.options]] package. This
  * primarily consists of:
  *   - [[FirrtlStage]], the internal and external (command line) interface to the FIRRTL compiler
  *   - A number of [[options.Phase Phase]]s that support and compartmentalize the individual operations of
  *     [[FirrtlStage]]
  *   - [[FirrtlOptions]], a class representing options that are necessary to drive the [[FirrtlStage]] and its
  *     [[firrtl.options.Phase Phase]]s
  *   - [[FirrtlOptionsView]], a utility that constructs an [[options.OptionsView OptionsView]] of [[FirrtlOptions]]
  *     from an [[AnnotationSeq]]
  *   - [[FirrtlCli]], the command line options that the [[FirrtlStage]] supports
  *   - [[FirrtlStageUtils]] containing miscellaneous utilities for [[stage]]
  */
package object stage {
  implicit object FirrtlOptionsView extends OptionsView[FirrtlOptions] {

    /**
      * @todo custom transforms are appended as discovered, can this be prepended safely?
      */
    def view(options: AnnotationSeq): FirrtlOptions = options
      .collect { case a: FirrtlOption => a }
      .foldLeft(new FirrtlOptions()){ (c, x) =>
        x match {
          case OutputFileAnnotation(f)           => c.copy(outputFileName = Some(f))
          case InfoModeAnnotation(i)             => c.copy(infoModeName = i)
          case CompilerAnnotation(cx)            => c.copy(compiler = cx)
          case FirrtlCircuitAnnotation(cir)      => c.copy(firrtlCircuit = Some(cir))
        }
      }
  }

  private [firrtl] implicit object FirrtlExecutionResultView extends OptionsView[FirrtlExecutionResult] {

    private lazy val dummyWriteEmitted = new WriteEmitted

    def view(options: AnnotationSeq): FirrtlExecutionResult = {
      val fopts = Viewer[FirrtlOptions].view(options)
      val emittedRes = options
        .collect{ case DeletedAnnotation(dummyWriteEmitted.name, a: EmittedAnnotation[_]) => a.value.value }
        .mkString("\n")

      options.collectFirst{ case a: FirrtlCircuitAnnotation => a.circuit } match {
        case None => FirrtlExecutionFailure("No circuit found in AnnotationSeq!")
        case Some(a) => FirrtlExecutionSuccess(
          emitType = fopts.compiler.getClass.getName,
          emitted = emittedRes,
          circuitState = CircuitState(
            circuit = a,
            form = fopts.compiler.outputForm,
            annotations = options,
            renames = None
          ))
      }
    }
  }

}
