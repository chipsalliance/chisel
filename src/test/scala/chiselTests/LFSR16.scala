// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
import chisel3.util.random.{PRNG, LFSR}

/**
  * This test creates two 4 sided dice.
  * Each cycle it adds them together and adds a count to the bin corresponding to that value
  * The asserts check that the bins show the correct distribution.
  */
//scalastyle:off magic.number
class LFSRDistribution(gen: => UInt, cycles: Int = 10000) extends BasicTester {

  val rv = gen
  val bins = Reg(Vec(8, UInt(32.W)))

  // Use tap points on each LFSR so values are more independent
  val die0 = Cat(Seq.tabulate(2) { i => rv(i) })
  val die1 = Cat(Seq.tabulate(2) { i => rv(i + 2) })

  val (trial, done) = Counter(true.B, cycles)

  val rollValue = die0 +& die1  // Note +& is critical because sum will need an extra bit.

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

class LFSRMaxPeriod(gen: => UInt) extends BasicTester {

  val rv = gen
  val started = RegNext(true.B, false.B)
  val seed = withReset(!started) { RegInit(rv) }

  val (_, wrap) = Counter(started, math.pow(2.0, rv.getWidth).toInt - 1)

  when (rv === seed && started) {
    chisel3.assert(wrap)
    stop()
  }

  val last = RegNext(rv)
  chisel3.assert(rv =/= last, "LFSR last value (0b%b) was equal to current value (0b%b)", rv, last)

}

/** Check that the output of the new LFSR is the same as the old LFSR */
class MeetTheNewLFSR16SameAsTheOldLFSR16 extends BasicTester {
  val en = Counter(2).value.asBool

  /** This is the exact implementation of the old LFSR16 algorithm */
  val oldLfsr = {
    val width = 16
    val lfsr = RegInit(1.U(width.W))
    when (en) {
      lfsr := Cat(lfsr(0)^lfsr(2)^lfsr(3)^lfsr(5), lfsr(width-1,1))
    }
    lfsr
  }

  /** The new LFSR16 uses equivalent taps and a reverse so that it can use LFSR(16) under the hood. */
  val newLfsr = LFSR16(en)

  val (_, done) = Counter(true.B, 16)

  assert(oldLfsr === newLfsr)

  when (done) {
    stop()
  }

}

class LFSRSpec extends ChiselPropSpec {
  property("LFSR16 can be used to produce pseudo-random numbers, this tests the distribution") {
    assertTesterPasses{ new LFSRDistribution(LFSR16()) }
  }

  property("LFSR16 period tester, Period should 2^16 - 1") {
    assertTesterPasses{ new LFSRMaxPeriod(LFSR16()) }
  }

  property("New LFSR16 is the same as the old LFSR16") {
    assertTesterPasses{ new MeetTheNewLFSR16SameAsTheOldLFSR16 }
  }
}
