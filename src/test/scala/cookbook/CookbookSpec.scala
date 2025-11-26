// SPDX-License-Identifier: Apache-2.0

package cookbook

import chisel3._
import chisel3.util.Counter
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Tester for concise cookbook tests
  *
  * Provides a length of test after which the test will pass
  */
abstract class CookbookTester(length: Int) extends Module {
  require(length >= 0, "Simulation length must be non-negative!")

  val (cycle, done) = Counter(true.B, length + 1) // + 1 cycle because done is actually wrap
  when(done) { stop() }
}

abstract class CookbookSpec extends AnyFlatSpec with Matchers with ChiselSim
