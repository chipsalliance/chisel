package firrtl.backends.experimental.smt.random

import firrtl.options.Dependency
import firrtl.testutils.LeanTransformSpec

class InvalidToRandomSpec extends LeanTransformSpec(Seq(Dependency(InvalidToRandomPass))) {
  behavior.of("InvalidToRandomPass")

  val src1 =
    s"""
       |circuit Test:
       |  module Test:
       |    input a : UInt<2>
       |    output o : UInt<8>
       |    output o2 : UInt<8>
       |    output o3 : UInt<8>
       |
       |    o is invalid
       |
       |    when eq(a, UInt(3)):
       |      o <= UInt(5)
       |
       |    o2 is invalid
       |    node o2_valid = eq(a, UInt(2))
       |    when o2_valid:
       |      o2 <= UInt(7)
       |
       |    o3 is invalid
       |    o3 <= UInt(3)
       |""".stripMargin

  it should "model invalid signals as random" in {

    val circuit = compile(src1, List()).circuit
    //println(circuit.serialize)
    val result = circuit.serialize.split('\n').map(_.trim)

    // the condition should end up as a new node if it wasn't a reference already
    assert(result.contains("node _GEN_0_invalid_cond = not(eq(a, UInt<2>(\"h3\")))"))
    assert(result.contains("node o2_valid = eq(a, UInt<2>(\"h2\"))"))

    // every invalid results in a random statement
    assert(result.contains("rand _GEN_0_invalid : UInt<3> when _GEN_0_invalid_cond"))
    assert(result.contains("rand _GEN_1_invalid : UInt<3> when not(o2_valid)"))

    // the random value is conditionally assigned
    assert(result.contains("node _GEN_0 = mux(_GEN_0_invalid_cond, _GEN_0_invalid, UInt<3>(\"h5\"))"))
    assert(result.contains("node _GEN_1 = mux(not(o2_valid), _GEN_1_invalid, UInt<3>(\"h7\"))"))

    // expressions that are trivially valid do not get randomized
    assert(result.contains("o3 <= UInt<2>(\"h3\")"))
    val defRandCount = result.count(_.contains("rand "))
    assert(defRandCount == 2)
  }

}
