// See LICENSE for license details.

package cookbook

import chisel3._

/* ### How do I create a Reg of type Vec?
 *
 * For information, please see the API documentation
 * (https://chisel.eecs.berkeley.edu/api/index.html#chisel3.core.Vec)
 */
class RegOfVec extends CookbookTester(2) {
  // Reg of Vec of 32-bit UInts
  val regOfVec = Reg(Vec(4, UInt(32.W)))
  regOfVec(0) := 123.U // a couple of assignments
  regOfVec(2) := regOfVec(0)

  // Simple test (cycle comes from superclass)
  when (cycle === 2.U) { assert(regOfVec(2) === 123.U) }
}

class RegOfVecSpec  extends CookbookSpec {
  "RegOfVec" should "work" in {
    assertTesterPasses { new RegOfVec }
  }
}
