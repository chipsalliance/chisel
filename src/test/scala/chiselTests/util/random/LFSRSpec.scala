// See LICENSE for license details.

package chiselTests.util.random

import chisel3._
import chisel3.util.{Counter, Enum}
import chisel3.util.random._
import chisel3.testers.BasicTester

import chiselTests.{ChiselFlatSpec, LFSRDistribution, LFSRMaxPeriod}

import math.pow

class FooLFSR(val reduction: LFSRReduce, seed: Option[BigInt]) extends PRNG(4, seed) with LFSR {
  def delta(s: Seq[Bool]): Seq[Bool] = s
}

/** This tests that after reset an LFSR is not locked up. This manually sets the seed of the LFSR at run-time to the
  * value that would cause it to lock up. It then asserts reset. The next cycle it checks that the value is NOT the
  * locked up value.
  * @param gen an LFSR to test
  * @param lockUpValue the value that would lock up the LFSR
  */
class LFSRResetTester(gen: => LFSR, lockUpValue: BigInt) extends BasicTester {

  val lfsr = Module(gen)
  lfsr.io.seed.valid := false.B
  lfsr.io.seed.bits := DontCare
  lfsr.io.increment := true.B

  val (count, done) = Counter(true.B, 5)

  lfsr.io.seed.valid := count === 1.U
  lfsr.io.seed.bits := lockUpValue.U(lfsr.width.W).asBools
  lfsr.io.increment := true.B

  when (count === 2.U) {
    assert(lfsr.io.out.asUInt === lockUpValue.U, "LFSR is NOT locked up, but should be!")
  }

  lfsr.reset := count === 3.U

  when (count === 4.U) {
    assert(lfsr.io.out.asUInt =/= lockUpValue.U, "LFSR is locked up, but should not be after reset!")
  }

  when (done) {
    stop()
  }

}

class LFSRSpec extends ChiselFlatSpec {

  def periodCheck(gen: (Int, Set[Int], LFSRReduce) => PRNG, reduction: LFSRReduce, range: Range): Unit = {
    it should s"have a maximal period over a range of widths (${range.head} to ${range.last}) using ${reduction.getClass}" in {
      range
        .foreach{ width =>
          LFSR.tapsMaxPeriod(width).foreach{ taps =>
            info(s"""width $width okay using taps: ${taps.mkString(", ")}""")
            assertTesterPasses(new LFSRMaxPeriod(PRNG(gen(width, taps, reduction))))
          }
        }
    }
  }

  behavior of "LFSR"

  it should "throw an exception if initialized to a seed of zero for XOR configuration" in {
    { the [IllegalArgumentException] thrownBy elaborate(new FooLFSR(XOR, Some(0))) }
      .getMessage should include ("Seed cannot be zero")
  }

  it should "throw an exception if initialized to a seed of all ones for XNOR configuration" in {
    { the [IllegalArgumentException] thrownBy elaborate(new FooLFSR(XNOR, Some(15))) }
      .getMessage should include ("Seed cannot be all ones")
  }

  it should "reset correctly without a seed for XOR configuration" in {
    assertTesterPasses(new LFSRResetTester(new FooLFSR(XOR, None), 0))
  }

  it should "reset correctly without a seed for XNOR configuration" in {
    assertTesterPasses(new LFSRResetTester(new FooLFSR(XNOR, None), 15))
  }

  behavior of "MaximalPeriodGaloisLFSR"

  it should "throw an exception if no LFSR taps are known" in {
    { the [IllegalArgumentException] thrownBy elaborate(new MaxPeriodGaloisLFSR(787)) }
      .getMessage should include ("No max period LFSR taps stored for requested width")
  }

  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new GaloisLFSR(w, t, reduction=r), XOR, 2 to 16)
  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new GaloisLFSR(w, t, reduction=r), XNOR, 2 to 16)

  ignore should "have a sane distribution for larger widths" in {
    ((17 to 32) ++ Seq(64, 128, 256, 512, 1024, 2048, 4096))
      .foreach{ width =>
        info(s"width $width okay!")
        assertTesterPasses(new LFSRDistribution(LFSR(width), math.pow(2, 22).toInt))
      }
  }

  behavior of "MaximalPeriodFibonacciLFSR"

  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new FibonacciLFSR(w, t, reduction=r), XOR, 2 to 16)
  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new FibonacciLFSR(w, t, reduction=r), XNOR, 2 to 16)

  behavior of "LFSR maximal period taps"

  it should "contain all the expected widths" in {
    ((2 to 786) ++ Seq(1024, 2048, 4096)).foreach(LFSR.tapsMaxPeriod.keys should contain (_))
  }

}
