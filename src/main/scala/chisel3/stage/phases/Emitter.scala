// See LICENSE for license details.

package chisel3.stage.phases

import firrtl.{AnnotationSeq, EmittedFirrtlCircuit, EmittedFirrtlCircuitAnnotation}
import firrtl.annotations.DeletedAnnotation
import firrtl.options.{Phase, StageOptions}
import firrtl.options.Viewer.view

import chisel3.internal.firrtl.{Emitter => OldEmitter}
import chisel3.stage.{ChiselCircuitAnnotation, ChiselOptions}

import java.io.{File, FileWriter}

/** Emit a [[chisel3.stage.ChiselCircuitAnnotation ChiselCircuitAnnotation]] to a file if a
  * [[chisel3.stage.ChiselCircuitAnnotation ChiselCircuitAnnotation]] is present. A deleted
  * [[firrtl.EmittedFirrtlCircuitAnnotation EmittedFirrtlCircuitAnnotation]] is added.
  *
  * @todo This should be switched to support correct emission of multiple circuits to multiple files. The API should
  * likely mirror how the [[firrtl.stage.phases.Compiler phases.Compiler]] parses annotations into "global" annotations
  * and left-associative per-ciruict annotations.
  * @todo The use of the deleted [[firrtl.EmittedFirrtlCircuitAnnotation EmittedFirrtlCircuitAnnotation]] is a kludge to
  * provide some breadcrumbs such that the emitter CHIRRTL can be provided back to the old Driver. This should be
  * removed or a better solution developed.
  */
object Emitter extends Phase {

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val copts = view[ChiselOptions](annotations)
    val sopts = view[StageOptions](annotations)

    annotations.flatMap {
      case a: ChiselCircuitAnnotation if copts.outputFile.isDefined =>
        val file = new File(sopts.getBuildFileName(copts.outputFile.get, Some(".fir")))
        val emitted = OldEmitter.emit(a.circuit)
        val w = new FileWriter(file)
        w.write(emitted)
        w.close()
        val anno = EmittedFirrtlCircuitAnnotation(EmittedFirrtlCircuit(a.circuit.name, emitted, ".fir"))
        Seq(DeletedAnnotation(name, anno), a)
      case a => Seq(a)
    }
  }

}
