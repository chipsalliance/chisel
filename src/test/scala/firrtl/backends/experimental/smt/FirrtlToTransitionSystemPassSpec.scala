// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt

import firrtl.annotations.{CircuitTarget, PresetAnnotation}
import firrtl.options.Dependency
import firrtl.testutils.LeanTransformSpec

class FirrtlToTransitionSystemPassSpec extends LeanTransformSpec(Seq(Dependency(firrtl.backends.experimental.smt.FirrtlToTransitionSystem))) {
  behavior of "FirrtlToTransitionSystem"

  it should "support preset wires" in {
    // In order to give registers an initial wire, we use preset annotated resets.
    // When using a wire instead of an input (which has the advantage of working regardless of the
    // module hierarchy), we need to initialize it in order to get through the wire initialization check.
    // In Chisel this generates a node which needs to be removed.

    val src = """circuit ModuleAB :
      |  module ModuleAB :
      |    input clock : Clock
      |    node _T = asAsyncReset(UInt<1>("h0"))
      |    node preset = _T
      |    reg REG : UInt<1>, clock with :
      |      reset => (preset, UInt<1>("h0"))
      |    assert(clock, UInt(1), not(REG), "REG == 0")
      |""".stripMargin
    val anno = PresetAnnotation(CircuitTarget("ModuleAB").module("ModuleAB").ref("preset"))

    val result = compile(src, List(anno))
    val sys = result.annotations.collectFirst{ case TransitionSystemAnnotation(sys) => sys }.get
    assert(sys.states.head.init.isDefined)
  }
}
