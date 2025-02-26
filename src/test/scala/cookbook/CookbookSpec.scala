// SPDX-License-Identifier: Apache-2.0

package cookbook

import chisel3._
<<<<<<< HEAD:src/test/scala/cookbook/CookbookSpec.scala
import chisel3.util._
import chisel3.testers.BasicTester

import chiselTests.ChiselFlatSpec
||||||| parent of 62bdfce5 ([test] Remove unnecessary usages of BasicTester):src/test/scala-2/cookbook/CookbookSpec.scala
import chisel3.util.Counter
import chisel3.simulator.scalatest.ChiselSim
import chisel3.testers.BasicTester
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
=======
import chisel3.util.Counter
import chisel3.simulator.scalatest.ChiselSim
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
>>>>>>> 62bdfce5 ([test] Remove unnecessary usages of BasicTester):src/test/scala-2/cookbook/CookbookSpec.scala

/** Tester for concise cookbook tests
  *
  * Provides a length of test after which the test will pass
  */
abstract class CookbookTester(length: Int) extends BasicTester {
  require(length >= 0, "Simulation length must be non-negative!")

  val (cycle, done) = Counter(true.B, length + 1) // + 1 cycle because done is actually wrap
  when(done) { stop() }
}

abstract class CookbookSpec extends ChiselFlatSpec
