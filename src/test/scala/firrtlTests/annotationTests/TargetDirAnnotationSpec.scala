// See LICENSE for license details.

package firrtlTests
package annotationTests

import firrtlTests._
import firrtl._
import firrtl.stage.TargetDirAnnotation

/** Looks for [[TargetDirAnnotation]] */
class FindTargetDirTransform(expected: String) extends Transform {
  def inputForm = HighForm
  def outputForm = HighForm
  var foundTargetDir = false
  var run = false
  def execute(state: CircuitState): CircuitState = {
    run = true
    state.annotations.collectFirst {
      case TargetDirAnnotation(expected) =>
        foundTargetDir = true
    }
    state
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
    val findTargetDir = new FindTargetDirTransform(targetDir) // looks for the annotation

    val optionsManager = new ExecutionOptionsManager("TargetDir") with HasFirrtlOptions {
      commonOptions = commonOptions.copy(targetDirName = targetDir,
                                         topName = "Top")
      firrtlOptions = firrtlOptions.copy(compilerName = "high",
                                         firrtlSource = Some(input),
                                         customTransforms = Seq(findTargetDir))
    }
    Driver.execute(optionsManager)

    // Check that FindTargetDirTransform transform is run and finds the annotation
    findTargetDir.run should be (true)
    findTargetDir.foundTargetDir should be (true)

    // Delete created directory
    val dir = new java.io.File(targetDir)
    dir.exists should be (true)
    FileUtils.deleteDirectoryHierarchy("a") should be (true)
  }

  it should "NOT be available as an annotation when using a raw compiler" in {
    val findTargetDir = new FindTargetDirTransform(targetDir) // looks for the annotation
    val compiler = new VerilogCompiler
    val circuit = Parser.parse(input split "\n")
    compiler.compileAndEmit(CircuitState(circuit, HighForm), Seq(findTargetDir))

    // Check that FindTargetDirTransform does not find the annotation
    findTargetDir.run should be (true)
    findTargetDir.foundTargetDir should be (false)
  }
}
