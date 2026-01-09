// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ParameterizedModule(invert: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
  })
  if (invert) {
    io.out := !io.in
  } else {
    io.out := io.in
  }
}

/** A simple test to check Module deduplication doesn't affect correctness (two
  * modules with the same name but different contents aren't aliased). Doesn't
  * check that deduplication actually happens, though.
  */
class ParameterizedModuleTester() extends Module {
  val invert = Module(new ParameterizedModule(true))
  val noninvert = Module(new ParameterizedModule(false))

  invert.io.in := true.B
  noninvert.io.in := true.B
  assert(invert.io.out === false.B)
  assert(noninvert.io.out === true.B)

  stop()
}

class ParameterizedModuleSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "Different parameterized modules" should "have different behavior" in {
    simulate(new ParameterizedModuleTester())(RunUntilFinished(3))
  }
}
