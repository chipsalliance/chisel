// See LICENSE for license details.

package chisel3

import firrtl._
import firrtl.annotations.DeletedAnnotation
import firrtl.options.OptionsView
import firrtl.options.Viewer
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlOptions}

import chisel3.internal.firrtl.Circuit
import chisel3.stage.ChiselOptions
import chisel3.stage.phases.{Convert, Emitter}

package object stage {

  implicit object ChiselOptionsView extends OptionsView[ChiselOptions] {

    def view(options: AnnotationSeq): ChiselOptions = options
      .collect { case a: ChiselOption => a }
      .foldLeft(new ChiselOptions()){ (c, x) =>
        x match {
          case _: NoRunFirrtlCompilerAnnotation.type => c.copy(runFirrtlCompiler = false)
          case _: PrintFullStackTraceAnnotation.type => c.copy(printFullStackTrace = true)
          case ChiselOutputFileAnnotation(f)         => c.copy(outputFile = Some(f))
          case ChiselCircuitAnnotation(a)            => c.copy(chiselCircuit = Some(a))
        }
      }

  }

  /** Construct a view of a [[firrtl.FirrtlExecutionResult FirrtlExecutionResult]]. This is not supposed to be used except
    * to enable the Driver compatibility layer.
    *
    * This is a straight copy-paste of the equivalent FIRRTL implicit object. This is intentional as this object should
    * not be made public.
    */
  private [chisel3] implicit object FirrtlExecutionResultView extends OptionsView[FirrtlExecutionResult] {

    def view(options: AnnotationSeq): FirrtlExecutionResult = {
      val fopts = Viewer.view[FirrtlOptions](options)
      val emittedRes = options.collect{ case a: EmittedAnnotation[_] => a.value.value }.mkString("\n")

      options.collectFirst{ case a: FirrtlCircuitAnnotation => a.circuit } match {
        case None => FirrtlExecutionFailure("No circuit found in AnnotationSeq!")
        case Some(a) => FirrtlExecutionSuccess(
          emitType = fopts.compiler.getClass.getSimpleName,
          emitted = emittedRes,
          circuitState = CircuitState(
            circuit = a,
            form = fopts.compiler.outputForm,
            annotations = firrtl.stage.phases.Strip.transform(options),
            renames = None
          ))
      }
    }
  }

  private [chisel3] implicit object ChiselExecutionResultView extends OptionsView[ChiselExecutionResult] {

    def view(options: AnnotationSeq): ChiselExecutionResult = {
      var chiselCircuit: Option[Circuit] = None
      var chirrtlCircuit: Option[String] = None
      options.foreach {
        case DeletedAnnotation(Convert.name, ChiselCircuitAnnotation(a)) => chiselCircuit = Some(a)
        case DeletedAnnotation(Emitter.name, EmittedFirrtlCircuitAnnotation(EmittedFirrtlCircuit(_, a, _))) =>
          chirrtlCircuit = Some(a)
        case _ =>
      }

      val fResult = Viewer.view[FirrtlExecutionResult](options)

      (chiselCircuit, chirrtlCircuit) match {
        case (None, _)          => ChiselExecutionFailure("Failed to elaborate Chisel circuit")
        case (Some(_), None)    => ChiselExecutionFailure("Failed to convert Chisel circuit to FIRRTL")
        case (Some(a), Some(b)) => ChiselExecutionSuccess( Some(a), b, Some(fResult))
      }

    }

  }

}
