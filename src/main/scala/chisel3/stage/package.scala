// See LICENSE for license details.

package chisel3

import firrtl._
import firrtl.annotations.DeletedAnnotation
import firrtl.options.OptionsView

import chisel3.internal.firrtl.{Circuit => ChiselCircuit}
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

  private[chisel3] implicit object ChiselExecutionResultView extends OptionsView[ChiselExecutionResult] {

    lazy val dummyWriteEmitted = new firrtl.stage.phases.WriteEmitted
    lazy val dummyConvert = new Convert
    lazy val dummyEmitter = new Emitter

    def view(options: AnnotationSeq): ChiselExecutionResult = {
      var chiselCircuit: Option[ChiselCircuit] = None
      var chirrtlCircuit: Option[String] = None

      options.foreach {
        case DeletedAnnotation(dummyConvert.name, ChiselCircuitAnnotation(a)) => chiselCircuit = Some(a)
        case DeletedAnnotation(dummyEmitter.name, EmittedFirrtlCircuitAnnotation(EmittedFirrtlCircuit(_, a, _))) =>
          chirrtlCircuit = Some(a)
        case _ =>
      }

      val fResult = firrtl.stage.phases.DriverCompatibility.firrtlResultView(options)

      (chiselCircuit, chirrtlCircuit) match {
        case (None, _)          => ChiselExecutionFailure("Failed to elaborate Chisel circuit")
        case (Some(_), None)    => ChiselExecutionFailure("Failed to convert Chisel circuit to FIRRTL")
        case (Some(a), Some(b)) => ChiselExecutionSuccess( Some(a), b, Some(fResult))
      }

    }

  }

}
