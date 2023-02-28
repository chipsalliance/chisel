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
    val filesWritten = mutable.HashMap.empty[String, Annotation]
    val serializable: AnnotationSeq = annotations.toSeq.flatMap {
      case _: Unserializable => None
      case a: CustomFileEmission =>
        val filename = a.filename(annotations)
        val canonical = filename.getCanonicalPath()

        filesWritten.get(canonical) match {
          case None =>
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
            filesWritten(canonical) = a
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
        a.replacements(filename)
      case a => Some(a)
    }

    sopts.annotationFileOut match {
      case None =>
      case Some(file) =>
        val pw = new PrintWriter(sopts.getBuildFileName(file, Some(".anno.json")))
        pw.write(JsonProtocol.serialize(serializable))
        pw.close()
    }

    annotations
  }

}
