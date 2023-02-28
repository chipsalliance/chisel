// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms

import firrtl.options.Dependency
import firrtl.testutils.LeanTransformSpec

class EnsureNamedStatementsSpec extends LeanTransformSpec(Seq(Dependency(EnsureNamedStatements))) {
  behavior.of("EnsureNamedStatements")

  it should "automatically name statements that do not have a name yet" in {
    val src = """circuit test :
                |  module test :
                |    input clock : Clock
                |    input stop_ : UInt<1>
                |    assert(clock, UInt(1), not(UInt(0)), "")
                |    stop(clock, UInt(1), 1) : stop_123
                |    stop(clock, UInt(1), 1)
                |    assert(clock, UInt(0), UInt(0), "")
                |    assume(clock, UInt(0), UInt(0), "")
                |    cover(clock, UInt(0), UInt(0), "")
                |    cover(clock, UInt(0), UInt(0), "")
                |
                |""".stripMargin

    val result = compile(src, List()).circuit.serialize.split('\n').map(_.trim)

    val expected = List(
      """assert(clock, UInt<1>("h1"), not(UInt<1>("h0")), "") : assert_0""",
      """stop(clock, UInt<1>("h1"), 1) : stop_123""",
      """stop(clock, UInt<1>("h1"), 1) : stop_0""",
      """assert(clock, UInt<1>("h0"), UInt<1>("h0"), "") : assert_1""",
      """assume(clock, UInt<1>("h0"), UInt<1>("h0"), "") : assume_0""",
      """cover(clock, UInt<1>("h0"), UInt<1>("h0"), "") : cover_0""",
      """cover(clock, UInt<1>("h0"), UInt<1>("h0"), "") : cover_1"""
    )
    expected.foreach(e => assert(result.contains(e)))
  }
}
