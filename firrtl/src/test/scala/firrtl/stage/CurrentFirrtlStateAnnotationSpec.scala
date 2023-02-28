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

  it should "produce an expected transform order for CHIRRTL -> Verilog" in {
    getTransforms("chirrtl") should contain(Dependency(firrtl.passes.CheckChirrtl))
  }

  it should "produce an expected transform order for minimum high FIRRTL -> Verilog" in {
    val transforms = getTransforms("mhigh")
    transforms should not contain noneOf(Dependency(firrtl.passes.CheckChirrtl), Dependency(firrtl.passes.InferTypes))
    transforms should contain(Dependency(firrtl.passes.CheckHighForm))
  }

  it should "produce an expected transform order for high FIRRTL -> Verilog" in {
    val transforms = getTransforms("high")
    transforms should not contain (Dependency[firrtl.transforms.DedupModules])
    (transforms should contain).allOf(
      Dependency(firrtl.passes.InferTypes),
      Dependency[firrtl.passes.ExpandWhensAndCheck]
    )
  }

  it should "produce an expected transform order for middle FIRRTL -> Verilog" in {
    val transforms = getTransforms("middle")
    transforms should not contain (Dependency[firrtl.passes.ExpandWhensAndCheck])
    (transforms should contain).allOf(Dependency(firrtl.passes.InferTypes), Dependency(firrtl.passes.LowerTypes))
  }

  it should "produce an expected transform order for low FIRRTL -> Verilog" in {
    val transforms = getTransforms("low")
    transforms should not contain (Dependency(firrtl.passes.LowerTypes))
    (transforms should contain).allOf(
      Dependency(firrtl.passes.InferTypes),
      Dependency(firrtl.passes.CommonSubexpressionElimination)
    )
  }

  it should "produce an expected transform order for optimized low FIRRTL -> Verilog" in {
    val transforms = getTransforms("low-opt")
    transforms should not contain (Dependency(firrtl.passes.CommonSubexpressionElimination))
    (transforms should contain).allOf(Dependency(firrtl.passes.InferTypes), Dependency[firrtl.transforms.VerilogRename])
  }

}
