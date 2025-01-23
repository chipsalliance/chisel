// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.random

import chisel3._
import circt.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.util.Counter
import chisel3.util.random.PRNG

import chiselTests.{ChiselFlatSpec, Utils}

class CyclePRNG(width: Int, seed: Option[BigInt], step: Int, updateSeed: Boolean)
    extends PRNG(width, seed, step, updateSeed) {

  def delta(s: Seq[Bool]): Seq[Bool] = s.last +: s.dropRight(1)

}

class PRNGStepTest extends BasicTester {

  val count2: UInt = Counter(true.B, 2)._1
  val count4: UInt = Counter(true.B, 4)._1

  val a: UInt = PRNG(new CyclePRNG(16, Some(1), 1, false), true.B)
  val b: UInt = PRNG(new CyclePRNG(16, Some(1), 2, false), count2 === 1.U)
  val c: UInt = PRNG(new CyclePRNG(16, Some(1), 4, false), count4 === 3.U)

  val (_, done) = Counter(true.B, 16)

  when(count2 === 0.U) {
    assert(a === b, "1-step and 2-step PRNGs did not agree! (0b%b != 0b%b)", a, b)
  }

  when(count4 === 0.U) {
    assert(a === c, "1-step and 4-step PRNGs did not agree!")
  }

  when(done) {
    stop()
  }

}

class PRNGUpdateSeedTest(updateSeed: Boolean, seed: BigInt, expected: BigInt) extends BasicTester {

  val a: CyclePRNG = Module(new CyclePRNG(16, Some(1), 1, updateSeed))

  val (count, done) = Counter(true.B, 4)

  a.io.increment := true.B
  a.io.seed.valid := count === 2.U
  a.io.seed.bits := seed.U(a.width.W).asBools

  when(count === 3.U) {
    assert(a.io.out.asUInt === expected.U, "Output didn't match!")
  }

  when(done) {
    stop()
  }

}

class PRNGSpec extends ChiselFlatSpec with Utils {

  behavior.of("PRNG")

  it should "throw an exception if the step size is < 1" in {
    {
      the[IllegalArgumentException] thrownBy extractCause[IllegalArgumentException] {
        ChiselStage.emitCHIRRTL(new CyclePRNG(0, Some(1), 1, true))
      }
    }.getMessage should include("Width must be greater than zero!")
  }

  it should "throw an exception if the step size is <= 0" in {
    {
      the[IllegalArgumentException] thrownBy extractCause[IllegalArgumentException] {
        ChiselStage.emitCHIRRTL(new CyclePRNG(1, Some(1), 0, true))
      }
    }.getMessage should include("Step size must be greater than one!")
  }

  it should "handle non-unary steps" in {
    assertTesterPasses(new PRNGStepTest)
  }

  it should "handle state update without and with updateSeed enabled" in {
    info("without updateSeed okay!")
    assertTesterPasses(new PRNGUpdateSeedTest(false, 3, 3))

    info("with updateSeed okay!")
    assertTesterPasses(new PRNGUpdateSeedTest(true, 3, 6))
  }

}
