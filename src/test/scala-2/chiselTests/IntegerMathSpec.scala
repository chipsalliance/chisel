// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import org.scalatest.propspec.AnyPropSpec

class IntegerMathTester extends Module {

  // TODO: Add more operators

  /* absolute values tests */

  val uint = 3.U(4.W)
  val sint = (-3).S
  val sintpos = 3.S
  val wrongSIntPos = 4.S

  assert(uint.abs === uint)
  assert(sint.abs === sintpos)
  assert(sintpos.abs === sintpos)

  assert(sint.abs =/= wrongSIntPos)

  stop()
}

class IntegerMathSpec extends AnyPropSpec with ChiselSim {
  property("All integer ops should return the correct result") {
    simulate { new IntegerMathTester }(RunUntilFinished(3))
  }
}
