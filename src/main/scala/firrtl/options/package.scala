// See LICENSE for license details.

package firrtl

package object options {

  implicit object StageOptionsView extends OptionsView[StageOptions] {
    def view(options: AnnotationSeq): StageOptions = options
      .collect { case a: StageOption => a }
      .foldLeft(new StageOptions())((c, x) =>
        x match {
          case TargetDirAnnotation(a) => c.copy(targetDir = a)
          /* Insert input files at the head of the Seq for speed and because order shouldn't matter */
          case InputAnnotationFileAnnotation(a) => c.copy(annotationFilesIn = a +: c.annotationFilesIn)
          case OutputAnnotationFileAnnotation(a) => c.copy(annotationFileOut = Some(a))
          /* Do NOT reorder program args. The order may matter. */
          case ProgramArgsAnnotation(a) => c.copy(programArgs = c.programArgs :+ a)
          case WriteDeletedAnnotation => c.copy(writeDeleted = true)
        }
      )
  }

}
