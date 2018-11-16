// See LICENSE for license details.

package firrtl

package object options {

  implicit object StageOptionsView extends OptionsView[StageOptions] {
    def view(options: AnnotationSeq): StageOptions = options
      .collect { case a: StageOption => a }
      .foldLeft(StageOptions())((c, x) =>
        x match {
          case TargetDirAnnotation(a) => c.copy(targetDir = a)
          case InputAnnotationFileAnnotation(a) => c.copy(annotationFiles = a +: c.annotationFiles)
        }
      )
  }

}
