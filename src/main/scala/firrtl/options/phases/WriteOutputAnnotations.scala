// See LICENSE for license details.

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.annotations.{DeletedAnnotation, JsonProtocol}
import firrtl.options.{Phase, StageOptions, Unserializable, Viewer}

import java.io.PrintWriter

/** [[firrtl.options.Phase Phase]] that writes an [[AnnotationSeq]] to a file. A file is written if and only if a
  * [[StageOptions]] view has a non-empty [[StageOptions.annotationFileOut annotationFileOut]].
  */
class WriteOutputAnnotations extends Phase {

  /** Write the input [[AnnotationSeq]] to a fie. */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val sopts = Viewer[StageOptions].view(annotations)
    val serializable = annotations.filter{
      case _: Unserializable    => false
      case _: DeletedAnnotation => sopts.writeDeleted
      case _                    => true
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
