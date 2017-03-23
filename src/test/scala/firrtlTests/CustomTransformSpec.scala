// See LICENSE for license details.

package firrtlTests

import firrtl.ir.Circuit
import firrtl._
import firrtl.passes.Pass
import firrtl.ir._

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
}

