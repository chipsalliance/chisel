// SPDX-License-Identifier: Apache-2.0

package chisel3

import firrtl._
import firrtl.options.OptionsView

import chisel3.internal.firrtl.{Circuit => ChiselCircuit}
import chisel3.stage.CircuitSerializationAnnotation.FirrtlFileFormat

package object stage {

  implicit object ChiselOptionsView extends OptionsView[ChiselOptions] {

    def view(options: AnnotationSeq): ChiselOptions = options.collect { case a: ChiselOption => a }
      .foldLeft(new ChiselOptions()) { (c, x) =>
        x match {
          case NoRunFirrtlCompilerAnnotation  => c.copy(runFirrtlCompiler = false)
          case PrintFullStackTraceAnnotation  => c.copy(printFullStackTrace = true)
          case ThrowOnFirstErrorAnnotation    => c.copy(throwOnFirstError = true)
<<<<<<< HEAD
          case WarnReflectiveNamingAnnotation => c.copy(warnReflectiveNaming = true)
=======
          case WarningsAsErrorsAnnotation     => c.copy(warningsAsErrors = true)
          case WarnReflectiveNamingAnnotation => c // Do nothing, ignored
>>>>>>> 49894666 (Add option to treat warnings as errors (#2676))
          case ChiselOutputFileAnnotation(f)  => c.copy(outputFile = Some(f))
          case ChiselCircuitAnnotation(a)     => c.copy(chiselCircuit = Some(a))
        }
      }

  }
}
