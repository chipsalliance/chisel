// See LICENSE for license details.

package firrtlTests
package annotationTests

import firrtlTests._
import firrtl._
import firrtl.annotations.{Annotation, NoTargetAnnotation}

case object FoundTargetDirTransformRanAnnotation extends NoTargetAnnotation
case object FoundTargetDirTransformFoundTargetDirAnnotation extends NoTargetAnnotation

/** Looks for [[TargetDirAnnotation]] */
class FindTargetDirTransform extends Transform {
  def inputForm = HighForm
  def outputForm = HighForm

  def execute(state: CircuitState): CircuitState = {
    val a: Option[Annotation] = state.annotations.collectFirst {
      case TargetDirAnnotation("a/b/c") => FoundTargetDirTransformFoundTargetDirAnnotation }
    state.copy(annotations = state.annotations ++ a ++ Some(FoundTargetDirTransformRanAnnotation))
  }
}

class TargetDirAnnotationSpec extends FirrtlFlatSpec {

  behavior of "The target directory"

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

    val optionsManager = new ExecutionOptionsManager("TargetDir") with HasFirrtlOptions {
      commonOptions = commonOptions.copy(targetDirName = targetDir,
                                         topName = "Top")
      firrtlOptions = firrtlOptions.copy(compilerName = "high",
                                         firrtlSource = Some(input),
                                         customTransforms = Seq(findTargetDir))
    }
    val annotations: Seq[Annotation] = Driver.execute(optionsManager) match {
      case a: FirrtlExecutionSuccess => a.circuitState.annotations
      case _ => fail
    }

    annotations should contain (FoundTargetDirTransformRanAnnotation)
    annotations should contain (FoundTargetDirTransformFoundTargetDirAnnotation)

    // Delete created directory
    val dir = new java.io.File(targetDir)
    dir.exists should be (true)
    FileUtils.deleteDirectoryHierarchy("a") should be (true)
  }

  it should "NOT be available as an annotation when using a raw compiler" in {
    val findTargetDir = new FindTargetDirTransform // looks for the annotation
    val compiler = new VerilogCompiler
    val circuit = Parser.parse(input split "\n")

    val annotations: Seq[Annotation] = compiler
      .compileAndEmit(CircuitState(circuit, HighForm), Seq(findTargetDir))
      .annotations

    // Check that FindTargetDirTransform does not find the annotation
    annotations should contain (FoundTargetDirTransformRanAnnotation)
    annotations should not contain (FoundTargetDirTransformFoundTargetDirAnnotation)
  }
}
