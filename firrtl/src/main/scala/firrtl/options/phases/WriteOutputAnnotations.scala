// SPDX-License-Identifier: Apache-2.0

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, JsonProtocol}
import firrtl.options.{
  BufferedCustomFileEmission,
  CustomFileEmission,
  Dependency,
  Phase,
  PhaseException,
  StageOptions,
  Unserializable,
  Viewer
}
import firrtl.options.internal.WriteableCircuitAnnotation

import java.io.{BufferedOutputStream, File, FileOutputStream, PrintWriter}

import scala.collection.mutable

/** [[firrtl.options.Phase Phase]] that writes an [[AnnotationSeq]] to the filesystem,
  *  according to the following rules:
  *  1) Annotations which extend [[CustomFileEmission]] are written seperately to their prescribed
  *     destinations and replaced per [[[CustomFileEmission.replacements replacements]].
  *  2) All remaining annotations are written to destination specified by
  *    [[StageOptions.annotationFileOut annotationFileOut]], iff the stage option is set, with the following exception:
  *    a) Annotations extending [[Unserializable]] are not written
  *    b) Annotations extending [[CustomFileEmission]] are written to the file they specify using the serialization they
  *    define.  They show up in the output Annotation file using their "replacements", if one is specified.
  */
class WriteOutputAnnotations extends Phase {

  override def prerequisites =
    Seq(Dependency[GetIncludes], Dependency[AddDefaults], Dependency[Checks])

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  /** Write the input [[AnnotationSeq]] to a fie. */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val sopts = Viewer[StageOptions].view(annotations)
    val filesToWrite = mutable.HashMap.empty[String, Annotation]
    // Grab the circuit annotation so we can write serializable annotations to it
    // We also must calculate the filename because the annotation will be deleted before calling
    // writeToFile
    var circuitAnnoOpt: Option[(File, WriteableCircuitAnnotation)] = None
    val serializable = annotations.flatMap { anno =>
      // Check for file clobbering
      anno match {
        case _: Unserializable =>
        case a: CustomFileEmission =>
          val filename = a.filename(annotations)
          val canonical = filename.getCanonicalPath()

          filesToWrite.get(canonical) match {
            case None =>
            case Some(first) =>
              val msg =
                s"""|Multiple CustomFileEmission annotations would be serialized to the same file, '$canonical'
                    |  - first writer:
                    |      class: ${first.getClass.getName}
                    |      trimmed serialization: ${first.serialize.take(80)}
                    |  - second writer:
                    |      class: ${a.getClass.getName}
                    |      trimmed serialization: ${a.serialize.take(80)}
                    |""".stripMargin
              throw new PhaseException(msg)
          }
          filesToWrite(canonical) = a
        case _ =>
      }
      // Write files with CustomFileEmission and filter out Unserializable ones
      anno match {
        case _:   Unserializable => None
        case wca: WriteableCircuitAnnotation =>
          val filename = wca.filename(annotations)
          if (circuitAnnoOpt.nonEmpty) {
            val Some((firstFN, firstWCA)) = circuitAnnoOpt
            val msg =
              s"""|Multiple circuit annotations found--only 1 is supported
                  | - first circuit:
                  |     filename: $firstFN
                  |     trimmed serialization: ${firstWCA.serialize.take(80)}
                  | - second circuit:
                  |     filename: $filename
                  |     trimmed serialization: ${wca.serialize.take(80)}
                  |
                  |""".stripMargin
            throw new PhaseException(msg)
          }
          circuitAnnoOpt = Some(filename -> wca)
          None
        case a: CustomFileEmission =>
          val filename = a.filename(annotations)
          val canonical = filename.getCanonicalPath()

          val w = new BufferedOutputStream(new FileOutputStream(filename))
          a match {
            // Further optimized emission
            case buf: BufferedCustomFileEmission =>
              val it = buf.getBytesBuffered
              it.foreach(bytearr => w.write(bytearr))
            // Regular emission
            case _ =>
              a.getBytes match {
                case arr: mutable.ArraySeq[Byte] => w.write(arr.array.asInstanceOf[Array[Byte]])
                case other => other.foreach(w.write(_))
              }
          }
          w.close()
          a.replacements(filename)
        case a => Some(a)
      }
    }

    // If the circuit annotation exists, write annotations to it
    circuitAnnoOpt match {
      case Some((file, circuitAnno)) =>
        circuitAnno.writeToFile(file, serializable)
      case None =>
        // Otherwise, write to the old .anno.json
        sopts.annotationFileOut match {
          case None =>
          case Some(file) =>
            val pw = new PrintWriter(sopts.getBuildFileName(file, Some(".anno.json")))
            pw.write(JsonProtocol.serialize(serializable))
            pw.close()
        }
    }

    annotations
  }

}
