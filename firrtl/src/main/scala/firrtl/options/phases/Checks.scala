// SPDX-License-Identifier: Apache-2.0

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.options.{OptionsException, OutputAnnotationFileAnnotation, Phase, TargetDirAnnotation}
import firrtl.options.Dependency

/** [[firrtl.options.Phase Phase]] that validates an [[AnnotationSeq]]. If successful, views of this [[AnnotationSeq]]
  * as [[StageOptions]] are guaranteed to succeed.
  */
class Checks extends Phase {

  override def prerequisites = Seq(Dependency[GetIncludes], Dependency[AddDefaults])

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  /** Validate an [[AnnotationSeq]] for [[StageOptions]]
    * @throws OptionsException if annotations are invalid
    */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {

    val td, outA = collection.mutable.ListBuffer[Annotation]()
    annotations.foreach {
      case a: TargetDirAnnotation            => td += a
      case a: OutputAnnotationFileAnnotation => outA += a
      case _ =>
    }

    if (td.size != 1) {
      val d = td.map { case TargetDirAnnotation(x) => x }
      throw new OptionsException(
        s"""|Exactly one target directory must be specified, but found `${d.mkString(", ")}` specified via:
            |    - explicit target directory: -td, --target-dir, TargetDirAnnotation
            |    - fallback default value""".stripMargin
      )
    }

    if (outA.size > 1) {
      val x = outA.map { case OutputAnnotationFileAnnotation(x) => x }
      throw new OptionsException(
        s"""|At most one output annotation file can be specified, but found '${x.mkString(", ")}' specified via:
            |    - an option or annotation: -foaf, --output-annotation-file, OutputAnnotationFileAnnotation""".stripMargin
      )
    }

    annotations
  }

}
