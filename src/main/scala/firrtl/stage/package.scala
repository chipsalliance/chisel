// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.annotations.DeletedAnnotation
import firrtl.options.OptionsView
import firrtl.stage.phases.WriteEmitted
import logger.LazyLogging

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
  implicit object FirrtlOptionsView extends OptionsView[FirrtlOptions] with LazyLogging {

    /**
      * @todo custom transforms are appended as discovered, can this be prepended safely?
      */
    def view(options: AnnotationSeq): FirrtlOptions = options.collect { case a: FirrtlOption => a }
      .foldLeft(new FirrtlOptions()) { (c, x) =>
        x match {
          case OutputFileAnnotation(f)      => c.copy(outputFileName = Some(f))
          case InfoModeAnnotation(i)        => c.copy(infoModeName = i)
          case FirrtlCircuitAnnotation(cir) => c.copy(firrtlCircuit = Some(cir))
          case a: CompilerAnnotation => logger.warn(s"Use of CompilerAnnotation is deprecated. Ignoring $a"); c
          case WarnNoScalaVersionDeprecation => c
          case PrettyNoExprInlining          => c
          case _: DisableFold => c
          case CurrentFirrtlStateAnnotation(a) => c
        }
      }
  }

  private[firrtl] implicit object FirrtlExecutionResultView
      extends OptionsView[FirrtlExecutionResult]
      with LazyLogging {

    def view(options: AnnotationSeq): FirrtlExecutionResult = {
      val emittedRes = options.collect { case a: EmittedAnnotation[_] => a.value.value }
        .mkString("\n")

      val emitters = options.collect { case RunFirrtlTransformAnnotation(e: Emitter) => e }
      if (emitters.length > 1) {
        logger.warn(
          "More than one emitter used which cannot be accurately represented" +
            "in the deprecated FirrtlExecutionResult: " + emitters.map(_.name).mkString(", ")
        )
      }
      val compilers = options.collect { case CompilerAnnotation(c) => c }
      val emitType = emitters.headOption.orElse(compilers.headOption).map(_.name).getOrElse("N/A")
      val form = emitters.headOption.orElse(compilers.headOption).map(_.outputForm).getOrElse(UnknownForm)

      options.collectFirst { case a: FirrtlCircuitAnnotation => a.circuit } match {
        case None => FirrtlExecutionFailure("No circuit found in AnnotationSeq!")
        case Some(a) =>
          FirrtlExecutionSuccess(
            emitType = emitType,
            emitted = emittedRes,
            circuitState = CircuitState(
              circuit = a,
              form = form,
              annotations = options,
              renames = None
            )
          )
      }
    }
  }

}
