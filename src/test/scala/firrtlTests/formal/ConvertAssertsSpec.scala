// SPDX-License-Identifier: Apache-2.0

package firrtlTests.formal

import firrtl._
import firrtl.testutils.FirrtlFlatSpec
import firrtl.transforms.formal.ConvertAsserts

class ConvertAssertsSpec extends FirrtlFlatSpec {
  val preamble =
    """circuit DUT:
      |  module DUT:
      |    input clock: Clock
      |    input reset: UInt<1>
      |    input x: UInt<8>
      |    output y: UInt<8>
      |    y <= x
      |    node ne5 = neq(x, UInt(5))
      |""".stripMargin

  "assert nodes" should "be converted to predicated prints and stops" in {
    val input = preamble +
      """    assert(clock, ne5, not(reset), "x should not equal 5")
        |""".stripMargin

    val ref = preamble +
      """    printf(clock, and(not(ne5), not(reset)), "x should not equal 5")
        |    stop(clock, and(not(ne5), not(reset)), 1)
        |""".stripMargin

    val outputCS = ConvertAsserts.execute(CircuitState(parse(input), Nil))
    (parse(outputCS.circuit.serialize)) should be(parse(ref))
  }

  "assert nodes with no message" should "omit printed messages" in {
    val input = preamble +
      """    assert(clock, ne5, not(reset), "")
        |""".stripMargin

    val ref = preamble +
      """    stop(clock, and(not(ne5), not(reset)), 1)
        |""".stripMargin

    val outputCS = ConvertAsserts.execute(CircuitState(parse(input), Nil))
    (parse(outputCS.circuit.serialize)) should be(parse(ref))
  }
}
