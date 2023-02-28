// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt.end2end

import firrtl.annotations.CircuitTarget
import firrtl.backends.experimental.smt.UninterpretedModuleAnnotation

class UninterpretedModulesSpec extends EndToEndSMTBaseSpec {

  private def testCircuit(assumption: String = ""): String = {
    s"""circuit UF00:
       |  module UF00:
       |    input clk: Clock
       |    input a: UInt<128>
       |    input b: UInt<128>
       |    input c: UInt<128>
       |
       |    inst m0 of Magic
       |    m0.a <= a
       |    m0.b <= b
       |
       |    inst m1 of Magic
       |    m1.a <= a
       |    m1.b <= c
       |
       |    assert(clk, eq(m0.r, m1.r), UInt(1), "m0.r == m1.r")
       |    $assumption
       |  extmodule Magic:
       |    input a: UInt<128>
       |    input b: UInt<128>
       |    output r: UInt<128>
       |""".stripMargin
  }
  private val magicAnno = UninterpretedModuleAnnotation(CircuitTarget("UF00").module("Magic"), "magic", 0)

  "two instances of the same uninterpreted module" should "give the same result when given the same inputs" taggedAs (RequiresZ3) in {
    val assumeTheSame = """assume(clk, eq(b,c), UInt(1), "b == c")"""
    test(testCircuit(assumeTheSame), MCSuccess, 1, "inputs are the same ==> outputs are the same", Seq(magicAnno))
  }
  "two instances of the same uninterpreted module" should "not always give the same result when given potentially different inputs" taggedAs (RequiresZ3) in {
    test(
      testCircuit(),
      MCFail(0),
      1,
      "inputs are not necessarily the same ==> outputs can be different",
      Seq(magicAnno)
    )
  }
}
