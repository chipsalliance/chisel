// See LICENSE for license details.

package firrtl.backends.experimental.smt.end2end

/** undefined values in firrtl are modelled as fresh auxiliary variables (inputs) */
class UndefinedFirrtlSpec extends EndToEndSMTBaseSpec {

  "division by zero" should "result in an arbitrary value" taggedAs (RequiresZ3) in {
    // the SMTLib spec defines the result of division by zero to be all 1s
    // https://cs.nyu.edu/pipermail/smt-lib/2015/000977.html
    def in(dEq: Int) =
      s"""circuit CC00:
         |  module CC00:
         |    input c: Clock
         |    input a: UInt<2>
         |    input b: UInt<2>
         |    assume(c, eq(b, UInt(0)), UInt(1), "b = 0")
         |    node d = div(a, b)
         |    assert(c, eq(d, UInt($dEq)), UInt(1), "d = $dEq")
         |""".stripMargin
    // we try to assert that (d = a / 0) is any fixed value which should be false
    (0 until 4).foreach { ii => test(in(ii), MCFail(0), 0, s"d = a / 0 = $ii") }
  }

  // TODO: rem should probably also be undefined, but the spec isn't 100% clear here

  "invalid signals" should "have an arbitrary values" taggedAs (RequiresZ3) in {
    def in(aEq: Int) =
      s"""circuit CC00:
         |  module CC00:
         |    input c: Clock
         |    wire a: UInt<2>
         |    a is invalid
         |    assert(c, eq(a, UInt($aEq)), UInt(1), "a = $aEq")
         |""".stripMargin
    // a should not be equivalent to any fixed value (0, 1, 2 or 3)
    (0 until 4).foreach { ii => test(in(ii), MCFail(0), 0, s"a = $ii") }
  }
}
