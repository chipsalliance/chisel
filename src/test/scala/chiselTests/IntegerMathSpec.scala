// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester

//scalastyle:off magic.number
class IntegerMathTester extends BasicTester {

  //TODO: Add more operators

  /* absolute values tests */

  val uint = 3.U(4.W)
  val sint = (-3).S
  val sintpos = 3.S
  val wrongSIntPos = 4.S

  assert(uint.abs() === uint)
  assert(sint.abs() === sintpos)
  assert(sintpos.abs() === sintpos)

  assert(sint.abs() =/= wrongSIntPos)

  stop()
}

class IntegerMathSpec extends ChiselPropSpec {
  property("All integer ops should return the correct result") {
    assertTesterPasses{ new IntegerMathTester }
  }
}
