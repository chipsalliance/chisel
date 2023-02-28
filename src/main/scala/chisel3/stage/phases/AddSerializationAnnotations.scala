// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase}
import firrtl.options.Viewer.view

import chisel3.stage._
import chisel3.stage.CircuitSerializationAnnotation._
import chisel3.ChiselException

/** Adds [[stage.CircuitSerializationAnnotation]]s based on [[ChiselOutputFileAnnotation]]
  */
class AddSerializationAnnotations extends Phase {

  override def prerequisites = Seq(Dependency[Elaborate], Dependency[AddImplicitOutputFile])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val chiselOptions = view[ChiselOptions](annotations)
    val circuit = chiselOptions.chiselCircuit.getOrElse {
      throw new ChiselException(
        s"Unable to locate the elaborated circuit, did ${classOf[Elaborate].getName} run correctly"
      )
    }
    val baseFilename = chiselOptions.outputFile.getOrElse(circuit.name)

    val (filename, format) = baseFilename match {
      case _ if baseFilename.endsWith(".pb") =>
        logger.warn(
          """[warn] Protobuf emission is deprecated in Chisel 3.6 and unsupported by the MFC. Change to an output file which does not include a ".pb" suffix. This behavior will change in Chisel 5 to emit FIRRTL text even if you provide a ".pb" suffix."""
        )
        (baseFilename.stripSuffix(".pb"), ProtoBufFileFormat)
      case _ => (baseFilename.stripSuffix(".fir"), FirrtlFileFormat)
    }

    CircuitSerializationAnnotation(circuit, filename, format) +: annotations
  }
}
