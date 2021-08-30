// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt.end2end

class AssertAssumeStopSpec extends EndToEndSMTBaseSpec {
  behavior.of("the SMT backend")

  private def prefix(ii: Int): String =
    s"""circuit AssertAssumStop$ii:
       |  module AssertAssumStop$ii:
       |    input clock: Clock
       |    input a: UInt<8>
       |    output b: UInt<8>
       |
       |    b <= add(a, UInt(16))
       |
       |""".stripMargin

  it should "support assertions" in {
    val src = prefix(0) +
      """    assert(clock, gt(b, a), lt(a, UInt(240)), "") : b_gt_a
        |""".stripMargin
    test(src, MCSuccess, kmax = 2)
  }

  it should "support assumptions" in {
    val src = prefix(1) +
      """    assert(clock, gt(b, a), UInt(1), "") : b_gt_a
        |""".stripMargin
    val srcWithAssume = prefix(2) +
      """    assert(clock, gt(b, a), UInt(1), "") : b_gt_a
        |    assume(clock, lt(a, UInt(240)), UInt(1), "") : a_lt_240
        |""".stripMargin
    // the assertion alone fails because of potential overflow
    test(src, MCFail(0), kmax = 2)
    // with the assumption that a is not too big, it works!
    test(srcWithAssume, MCSuccess, kmax = 2)
  }

  it should "treat stop with ret =/= 0 as assertion" in {
    val src = prefix(3) +
      """    stop(clock, not(gt(b, a)), 1) : b_gt_a
        |""".stripMargin
    val srcWithAssume = prefix(4) +
      """    stop(clock, not(gt(b, a)), 1) : b_gt_a
        |    assume(clock, lt(a, UInt(240)), UInt(1), "") : a_lt_240
        |""".stripMargin
    // the assertion alone fails because of potential overflow
    test(src, MCFail(0), kmax = 2)
    // with the assumption that a is not too big, it works!
    test(srcWithAssume, MCSuccess, kmax = 2)
  }

  it should "ignore stop with ret === 0" in {
    val src = prefix(5) +
      """    stop(clock, not(gt(b, a)), 1) : b_gt_a
        |    assume(clock, lt(a, UInt(240)), UInt(1), "") : a_lt_240
        |    stop(clock, UInt(1), 0) : always_stop
        |""".stripMargin
    test(src, MCSuccess, kmax = 2)
  }

}
