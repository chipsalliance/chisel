// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.{AnnotationSeq, EmittedModuleAnnotation, EmittedCircuitAnnotation}
import firrtl.options.{Phase, PreservesAll, StageOptions, Viewer}
import firrtl.stage.FirrtlOptions

import java.io.PrintWriter

/** [[firrtl.options.Phase Phase]] that writes any [[EmittedAnnotation]]s in an input [[AnnotationSeq]] to one or more
  * files. The input [[AnnotationSeq]] is viewed as both [[FirrtlOptions]] and [[firrtl.options.StageOptions
  * StageOptions]] to determine the output filenames in the following way:
  *   - [[EmittedModuleAnnotation]]s are written to a file in [[firrtl.options.StageOptions.targetDir
  *     StageOptions.targetDir]] with the same name as the module and the [[EmittedComponent.outputSuffix outputSuffix]]
  *     that the [[EmittedComponent]] specified
  *   - [[EmittedCircuitAnnotation]]s are written to a file in [[firrtl.options.StageOptions.targetDir
  *     StageOptions.targetDir]] using the [[FirrtlOptions.outputFileName]] viewed from the [[AnnotationSeq]]. If no
  *     [[FirrtlOptions.outputFileName]] exists, then the top module/main name will be used. The
  *     [[EmittedComponent.outputSuffix outputSuffix]] will be appended as needed.
  *
  * This does no sanity checking of the input [[AnnotationSeq]]. This simply writes any modules or circuits it sees to
  * files. If you need additional checking, then you should stack an appropriate checking phase before this.
  *
  * Any annotations written to files will be deleted.
  */
class WriteEmitted extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq.empty

  override val dependents = Seq.empty

  /** Write any [[EmittedAnnotation]]s in an [[AnnotationSeq]] to files. Written [[EmittedAnnotation]]s are deleted. */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val fopts = Viewer[FirrtlOptions].view(annotations)
    val sopts = Viewer[StageOptions].view(annotations)

    annotations.flatMap {
      case a: EmittedModuleAnnotation[_] =>
        val pw = new PrintWriter(sopts.getBuildFileName(a.value.name, Some(a.value.outputSuffix)))
        pw.write(a.value.value)
        pw.close()
        None
      case a: EmittedCircuitAnnotation[_] =>
        val pw = new PrintWriter(
          sopts.getBuildFileName(fopts.outputFileName.getOrElse(a.value.name), Some(a.value.outputSuffix)))
        pw.write(a.value.value)
        pw.close()
        None
      case a => Some(a)
    }

  }
}
