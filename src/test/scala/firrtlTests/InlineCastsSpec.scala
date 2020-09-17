// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.transforms.InlineCastsTransform
import firrtl.testutils.FirrtlFlatSpec

/*
 * Note: InlineCasts is still part of mverilog, so this test must both:
 * - Test that the InlineCasts fix is effective given the current mverilog
 * - Provide a test that will be robust if and when InlineCasts is no longer run in mverilog
 *
 * This is why the test passes InlineCasts as a custom transform: to future-proof it so that
 * it can do real LEC against no-InlineCasts. It currently is just a sanity check that the
 * emitted Verilog is legal, but it will automatically become a more meaningful test when
 * InlineCasts is not run in mverilog.
 */
class InlineCastsEquivalenceSpec extends FirrtlFlatSpec {
  "InlineCastsTransform" should "not produce broken Verilog" in {
    val input =
      s"""circuit literalsel_fir:
         |  module literalsel_fir:
         |    input i: UInt<4>
         |    output o: SInt<8>
         |    o <= pad(asSInt(UInt<2>("h1")), 8)
         |""".stripMargin
    firrtlEquivalenceTest(input, Seq(new InlineCastsTransform))
  }
}
