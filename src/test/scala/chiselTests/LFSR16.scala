// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

/**
  * This test creates two 4 sided dice.
  * Each cycle it adds them together and adds a count to the bin corresponding to that value
  * The asserts check that the bins show the correct distribution.
  */
//scalastyle:off magic.number
class LFSRTester extends BasicTester {
  val bins = Reg(Vec(8, UInt(32.W)))

  // Use tap points on each LFSR so values are more independent
  val die0 = Cat(Seq.tabulate(2) { i => LFSR16()(i) })
  val die1 = Cat(Seq.tabulate(2) { i => LFSR16()(i + 2) })

  val (trial, done) = Counter(true.B, 10000)

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

class LFSR16MaxPeriod extends BasicTester {

  val lfsr16 = LFSR16()
  val started = RegNext(true.B, false.B)

  val (_, wrap) = Counter(started, math.pow(2.0, 16).toInt - 1)
  when (lfsr16 === 1.U && started) {
    chisel3.assert(wrap)
    stop()
  }
}

class LFSRSpec extends ChiselPropSpec {
  property("LFSR16 can be used to produce pseudo-random numbers, this tests the distribution") {
    assertTesterPasses{ new LFSRTester }
  }

  property("LFSR16 period tester, Period should 2^16 - 1") {
    assertTesterPasses { new LFSR16MaxPeriod }
  }
}
