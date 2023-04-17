// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, StageOptions}
import firrtl.options.Viewer.view

import chisel3.stage.{ChiselCircuitAnnotation, ChiselOptions, CircuitSerializationAnnotation}
import CircuitSerializationAnnotation.FirrtlFileFormat

import java.io.{BufferedOutputStream, File, FileOutputStream}

/** Emit a [[chisel3.stage.ChiselCircuitAnnotation]] to a file if a [[chisel3.stage.ChiselOutputFileAnnotation]] is
  * present.
  *
  * @todo This should be switched to support correct emission of multiple circuits to multiple files.
  */
class Emitter extends Phase {

  override def prerequisites =
    Seq(
      Dependency[Elaborate],
      Dependency[AddImplicitOutputFile],
      Dependency[AddImplicitOutputAnnotationFile],
      Dependency[MaybeAspectPhase]
    )
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[Convert])
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val copts = view[ChiselOptions](annotations)
    val sopts = view[StageOptions](annotations)

    annotations.flatMap {
      case a: ChiselCircuitAnnotation if copts.outputFile.isDefined =>
        val filename = sopts.getBuildFileName(copts.outputFile.get, Some(".fir"))
        val csa = CircuitSerializationAnnotation(a.circuit, filename, FirrtlFileFormat)
        csa.doWriteToFile(new File(filename), Nil)
        Some(a)
      case a => Some(a)
    }
    annotations
  }

}
