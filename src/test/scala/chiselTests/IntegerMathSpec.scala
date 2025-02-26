// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
<<<<<<< HEAD:src/test/scala/chiselTests/IntegerMathSpec.scala
import chisel3.testers.BasicTester
||||||| parent of 62bdfce5 ([test] Remove unnecessary usages of BasicTester):src/test/scala-2/chiselTests/IntegerMathSpec.scala
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testers.BasicTester
import org.scalatest.propspec.AnyPropSpec
=======
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import org.scalatest.propspec.AnyPropSpec
>>>>>>> 62bdfce5 ([test] Remove unnecessary usages of BasicTester):src/test/scala-2/chiselTests/IntegerMathSpec.scala

class IntegerMathTester extends Module {

  //TODO: Add more operators

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

class IntegerMathSpec extends ChiselPropSpec {
  property("All integer ops should return the correct result") {
    assertTesterPasses { new IntegerMathTester }
  }
}
