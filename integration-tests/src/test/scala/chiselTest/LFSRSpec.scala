// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.random

import chisel3._
import circt.stage.ChiselStage
import chisel3.util.{Cat, Counter}
import chisel3.util.random._
import chisel3.testers.{BasicTester, TesterDriver}
import chiselTests.{ChiselFlatSpec, Utils}

class FooLFSR(val reduction: LFSRReduce, seed: Option[BigInt]) extends PRNG(4, seed) with LFSR {
  def delta(s: Seq[Bool]): Seq[Bool] = s
}

class LFSRMaxPeriod(gen: => UInt) extends BasicTester {

  val rv = gen
  val started = RegNext(true.B, false.B)
  val seed = withReset(!started) { RegInit(rv) }

  val (_, wrap) = Counter(started, math.pow(2.0, rv.getWidth).toInt - 1)

  when(rv === seed && started) {
    chisel3.assert(wrap)
    stop()
  }

  val last = RegNext(rv)
  chisel3.assert(rv =/= last, "LFSR last value (0b%b) was equal to current value (0b%b)", rv, last)

}

/**
  * This test creates two 4 sided dice.
  * Each cycle it adds them together and adds a count to the bin corresponding to that value
  * The asserts check that the bins show the correct distribution.
  */
class LFSRDistribution(gen: => UInt, cycles: Int = 10000) extends BasicTester {

  val rv = gen
  val bins = Reg(Vec(8, UInt(32.W)))

  // Use tap points on each LFSR so values are more independent
  val die0 = Cat(Seq.tabulate(2) { i => rv(i) })
  val die1 = Cat(Seq.tabulate(2) { i => rv(i + 2) })

  val (trial, done) = Counter(true.B, cycles)

  val rollValue = die0 +& die1 // Note +& is critical because sum will need an extra bit.

  bins(rollValue) := bins(rollValue) + 1.U

  when(done) {
    printf(p"bins: $bins\n") // Note using the printable interpolator p"" to print out a Vec

    // test that the distribution feels right.
    assert(bins(1) > bins(0))
    assert(bins(2) > bins(1))
    assert(bins(3) > bins(2))
    assert(bins(4) < bins(3))
    assert(bins(5) < bins(4))
    assert(bins(6) < bins(5))
    assert(bins(7) === 0.U)

    stop()
  }
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

  when(count === 2.U) {
    assert(lfsr.io.out.asUInt === lockUpValue.U, "LFSR is NOT locked up, but should be!")
  }

  lfsr.reset := count === 3.U

  when(count === 4.U) {
    assert(lfsr.io.out.asUInt =/= lockUpValue.U, "LFSR is locked up, but should not be after reset!")
  }

  when(done) {
    stop()
  }

}

class LFSRSpec extends ChiselFlatSpec with Utils {

  def periodCheck(gen: (Int, Set[Int], LFSRReduce) => PRNG, reduction: LFSRReduce, range: Range): Unit = {
    val testName = s"have a maximal period over a range of widths (${range.head} to ${range.last})" +
      s" using ${reduction.getClass}"
    //TODO: SFC->MFC, these tests fail due to a bootstrap problem under MFC in LFSRMaxPeriod
    it should testName ignore {
      range.foreach { width =>
        LFSR.tapsMaxPeriod(width).foreach { taps =>
          info(s"""width $width okay using taps: ${taps.mkString(", ")}""")
          assertTesterPasses(
            new LFSRMaxPeriod(PRNG(gen(width, taps, reduction))),
            annotations = TesterDriver.verilatorOnly
          )
        }
      }
    }
  }

  behavior.of("LFSR")

  it should "throw an exception if initialized to a seed of zero for XOR configuration" in {
    {
      the[IllegalArgumentException] thrownBy extractCause[IllegalArgumentException] {
        ChiselStage.emitCHIRRTL(new FooLFSR(XOR, Some(0)))
      }
    }.getMessage should include("Seed cannot be zero")
  }

  it should "throw an exception if initialized to a seed of all ones for XNOR configuration" in {
    {
      the[IllegalArgumentException] thrownBy extractCause[IllegalArgumentException] {
        ChiselStage.emitCHIRRTL(new FooLFSR(XNOR, Some(15)))
      }
    }.getMessage should include("Seed cannot be all ones")
  }

  it should "reset correctly without a seed for XOR configuration" in {
    assertTesterPasses(new LFSRResetTester(new FooLFSR(XOR, None), 0))
  }

  it should "reset correctly without a seed for XNOR configuration" in {
    assertTesterPasses(new LFSRResetTester(new FooLFSR(XNOR, None), 15))
  }

  behavior.of("MaximalPeriodGaloisLFSR")

  it should "throw an exception if no LFSR taps are known" in {
    {
      the[IllegalArgumentException] thrownBy extractCause[IllegalArgumentException] {
        ChiselStage.emitCHIRRTL(new MaxPeriodGaloisLFSR(787))
      }
    }.getMessage should include("No max period LFSR taps stored for requested width")
  }

  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new GaloisLFSR(w, t, reduction = r), XOR, 2 to 16)
  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new GaloisLFSR(w, t, reduction = r), XNOR, 2 to 16)

  ignore should "have a sane distribution for larger widths" in {
    ((17 to 32) ++ Seq(64, 128, 256, 512, 1024, 2048, 4096)).foreach { width =>
      info(s"width $width okay!")
      assertTesterPasses(new LFSRDistribution(LFSR(width), math.pow(2, 22).toInt))
    }
  }

  behavior.of("MaximalPeriodFibonacciLFSR")

  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new FibonacciLFSR(w, t, reduction = r), XOR, 2 to 16)
  periodCheck((w: Int, t: Set[Int], r: LFSRReduce) => new FibonacciLFSR(w, t, reduction = r), XNOR, 2 to 16)

  behavior.of("LFSR maximal period taps")

  it should "contain all the expected widths" in {
    ((2 to 786) ++ Seq(1024, 2048, 4096)).foreach(LFSR.tapsMaxPeriod.keys should contain(_))
  }

}
