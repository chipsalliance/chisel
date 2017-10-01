// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir.Circuit
import firrtl.Parser.IgnoreInfo
import firrtl.passes._
import firrtl.transforms._

class ReplaceAccessesSpec extends FirrtlFlatSpec {
  val transforms = Seq(
    ToWorkingIR,
    ResolveKinds,
    InferTypes,
    ResolveGenders,
    InferWidths,
    ReplaceAccesses)
  protected def exec(input: String) = {
    transforms.foldLeft(CircuitState(parse(input), UnknownForm)) {
      (c: CircuitState, t: Transform) => t.runTransform(c)
    }.circuit.serialize
  }
}

class ReplaceAccessesMultiDim extends ReplaceAccessesSpec {
  "ReplacesAccesses" should "replace constant accesses with fixed indices" in {
    val input =
      """circuit Top :
  module Top :
    input clock : Clock
    output out : UInt<1>
    reg r_vec : UInt<1>[4][2], clock
    out <= r_vec[UInt<2>(2)][UInt<1>(1)]
"""
    val check =
      """circuit Top :
  module Top :
    input clock : Clock
    output out : UInt<1>
    reg r_vec : UInt<1>[4][2], clock with :
      reset => (UInt<1>(0), r_vec)
    out <= r_vec[2][1]
"""
    (parse(exec(input))) should be (parse(check))
  }
}
