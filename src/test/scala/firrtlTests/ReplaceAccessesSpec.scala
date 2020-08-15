// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.passes._
import firrtl.testutils._

class ReplaceAccessesSpec extends FirrtlFlatSpec {
  val transforms = Seq(ToWorkingIR, ResolveKinds, InferTypes, ResolveFlows, new InferWidths, ReplaceAccesses)
  protected def exec(input: String) = {
    transforms
      .foldLeft(CircuitState(parse(input), UnknownForm)) { (c: CircuitState, t: Transform) =>
        t.runTransform(c)
      }
      .circuit
      .serialize
  }
}

class ReplaceAccessesMultiDim extends ReplaceAccessesSpec {
  "ReplacesAccesses" should "replace constant accesses with fixed indices" in {
    val input =
      """circuit Top :
  module Top :
    input clock : Clock
    output out : UInt<1>
    reg r_vec : UInt<1>[4][3], clock
    out <= r_vec[UInt<2>(2)][UInt<1>(1)]
"""
    val check =
      """circuit Top :
  module Top :
    input clock : Clock
    output out : UInt<1>
    reg r_vec : UInt<1>[4][3], clock with :
      reset => (UInt<1>(0), r_vec)
    out <= r_vec[2][1]
"""
    (parse(exec(input))) should be(parse(check))
  }

  "ReplacesAccesses" should "NOT generate out-of-bounds indices" in {
    val input =
      """circuit Top :
  module Top :
    input clock : Clock
    output out : UInt<1>
    reg r_vec : UInt<1>[4][2], clock
    out <= r_vec[UInt<3>(1)][UInt<3>(8)]
"""
    val check =
      """circuit Top :
  module Top :
    input clock : Clock
    output out : UInt<1>
    reg r_vec : UInt<1>[4][2], clock with :
      reset => (UInt<1>(0), r_vec)
    out <= r_vec[1][UInt<3>(8)]
"""
    (parse(exec(input))) should be(parse(check))
  }
}
