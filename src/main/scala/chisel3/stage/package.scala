// SPDX-License-Identifier: Apache-2.0

package chisel3

import firrtl._
import firrtl.options.OptionsView

import chisel3.internal.firrtl.{Circuit => ChiselCircuit}
import chisel3.stage.CircuitSerializationAnnotation.FirrtlFileFormat

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
}
