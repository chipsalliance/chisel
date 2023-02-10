// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl.options.Dependency
import firrtl.stage.transforms.Compiler
import firrtl.stage.TransformManager.TransformDependency
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CurrentFirrtlStateAnnotationSpec extends AnyFlatSpec with Matchers {

  def getTransforms(input: String): Seq[TransformDependency] = {
    val currentState = CurrentFirrtlStateAnnotation
      .options(0)
      .toAnnotationSeq(input)
      .collectFirst {
        case CurrentFirrtlStateAnnotation(currentState) => currentState
      }
      .get
    new Compiler(Forms.VerilogOptimized, currentState).flattenedTransformOrder.map(Dependency.fromTransform)
  }

  behavior.of("CurrentFirrtlStateAnnotation")

  it should "produce an expected transform order for low FIRRTL -> Verilog" in {
    val transforms = getTransforms("low")
    transforms should contain(Dependency(firrtl.passes.CommonSubexpressionElimination))
  }

  it should "produce an expected transform order for optimized low FIRRTL -> Verilog" in {
    val transforms = getTransforms("low-opt")
    transforms should not contain (Dependency(firrtl.passes.CommonSubexpressionElimination))
  }

}
