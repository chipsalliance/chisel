// See LICENSE for license details.

package firrtlTests

import firrtl.ir.Circuit
import firrtl._
import firrtl.passes.Pass
import firrtl.ir._
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.stage.{FirrtlSourceAnnotation, FirrtlStage, RunFirrtlTransformAnnotation}

class CustomTransformSpec extends FirrtlFlatSpec {
  behavior of "Custom Transforms"

  they should "be able to introduce high firrtl" in {
    // Simple module
    val delayModuleString = """
      |circuit Delay :
      |  module Delay :
      |    input clock : Clock
      |    input reset : UInt<1>
      |    input a : UInt<32>
      |    input en : UInt<1>
      |    output b : UInt<32>
      |
      |    reg r : UInt<32>, clock
      |    r <= r
      |    when en :
      |      r <= a
      |    b <= r
      |""".stripMargin
    val delayModuleCircuit = parse(delayModuleString)
    val delayModule = delayModuleCircuit.modules.find(_.name == delayModuleCircuit.main).get

    class ReplaceExtModuleTransform extends SeqTransform {
      class ReplaceExtModule extends Pass {
        def run(c: Circuit): Circuit = c.copy(
          modules = c.modules map {
            case ExtModule(_, "Delay", _, _, _) => delayModule
            case other => other
          }
        )
      }
      def transforms = Seq(new ReplaceExtModule)
      def inputForm = LowForm
      def outputForm = HighForm
    }

    runFirrtlTest("CustomTransform", "/features", customTransforms = List(new ReplaceExtModuleTransform))
  }

  they should "not cause \"Internal Errors\"" in {
    val input = """
      |circuit test :
      |  module test :
      |    output out : UInt
      |    out <= UInt(123)""".stripMargin
    val errorString = "My Custom Transform failed!"
    class ErroringTransform extends Transform {
      def inputForm = HighForm
      def outputForm = HighForm
      def execute(state: CircuitState): CircuitState = {
        require(false, errorString)
        state
      }
    }
    val optionsManager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
      firrtlOptions = FirrtlExecutionOptions(
        firrtlSource = Some(input),
        customTransforms = List(new ErroringTransform))
    }
    (the [java.lang.IllegalArgumentException] thrownBy {
      Driver.execute(optionsManager)
    }).getMessage should include (errorString)
  }

  object Foo {
    class A extends Transform {
      def inputForm = HighForm
      def outputForm = HighForm
      def execute(s: CircuitState) = {
        println(name)
        s
      }
    }
  }

  they should "work if placed inside an object" in {
    val input =
      """|circuit Foo:
         |  module Foo:
         |    node a = UInt<1>(0)
         |""".stripMargin
    val annotations = Seq(
      RunFirrtlTransformAnnotation(new Foo.A),
      FirrtlSourceAnnotation(input)
    )
    (new FirrtlStage).execute(Array.empty, annotations)
  }
}
