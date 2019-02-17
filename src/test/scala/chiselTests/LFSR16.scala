// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

/**
  * This test creates two 4 sided dice.
  * each cycle it adds them together and adds a count to the bin corresponding to that value
  * The asserts check that the bins show the correct distribution.
  */
//scalastyle:off magic.number
class LSFRTester extends BasicTester {
  val bins = Reg(Vec(8, UInt(32.W)))

  // Use tap points on each LFSR so values are more independent
  val die0 = Cat(Seq.tabulate(2) { i => LFSR16()(i) })
  val die1 = Cat(Seq.tabulate(2) { i => LFSR16()(i + 2) })

  val (trial, done) = Counter(true.B, 100000)

  val rollValue = die0 +& die1  // Note +& is critical because sum will need an extra bit.

  bins(rollValue) := bins(rollValue) + 1.U

  when(done) {
    printf("bins: %d %d %d %d %d %d %d %d\n",
      bins(0), bins(1), bins(2), bins(3), bins(4), bins(5), bins(6), bins(7))

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

class LFSRSpec extends ChiselPropSpec {
  property("LFSR16 can be used to produce pseudo-random numbers, this tests the distribution") {
    assertTesterPasses{ new LSFRTester }
  }
}
