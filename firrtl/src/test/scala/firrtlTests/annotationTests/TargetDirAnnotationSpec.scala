// SPDX-License-Identifier: Apache-2.0

package firrtlTests
package annotationTests

import firrtl._
import firrtl.testutils.FirrtlFlatSpec
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlStage, RunFirrtlTransformAnnotation}

case object FoundTargetDirTransformRanAnnotation extends NoTargetAnnotation
case object FoundTargetDirTransformFoundTargetDirAnnotation extends NoTargetAnnotation

/** Looks for [[TargetDirAnnotation]] */
class FindTargetDirTransform extends Transform {
  def inputForm = HighForm
  def outputForm = HighForm

  def execute(state: CircuitState): CircuitState = {
    val a: Option[Annotation] = state.annotations.collectFirst {
      case TargetDirAnnotation("a/b/c") => FoundTargetDirTransformFoundTargetDirAnnotation
    }
    state.copy(annotations = state.annotations ++ a ++ Some(FoundTargetDirTransformRanAnnotation))
  }
}

class TargetDirAnnotationSpec extends FirrtlFlatSpec {

  behavior.of("The target directory")

  val input =
    """circuit Top :
      |  module Top :
      |    input foo : UInt<32>
      |    output bar : UInt<32>
      |    bar <= foo
      """.stripMargin
  val targetDir = "a/b/c"

  it should "be available as an annotation when using execution options" in {
    val findTargetDir = new FindTargetDirTransform // looks for the annotation

    val annotations: Seq[Annotation] = (new FirrtlStage).execute(
      Array("--target-dir", targetDir, "--compiler", "high"),
      Seq(
        FirrtlCircuitAnnotation(Parser.parse(input)),
        RunFirrtlTransformAnnotation(findTargetDir)
      )
    )

    annotations should contain(FoundTargetDirTransformRanAnnotation)
    annotations should contain(FoundTargetDirTransformFoundTargetDirAnnotation)

    // Delete created directory
    val dir = new java.io.File(targetDir)
    dir.exists should be(true)
    FileUtils.deleteDirectoryHierarchy("a") should be(true)
  }
}
