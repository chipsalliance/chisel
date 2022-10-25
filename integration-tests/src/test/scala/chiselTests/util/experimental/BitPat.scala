// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import chisel3._
import chiseltest._
import chiseltest.formal._
import scala.util.Random
import org.scalatest.flatspec.AnyFlatSpec
import chisel3.util.experimental.BitSet

class BitSetRangeTestModule(start: BigInt, length: BigInt, width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  val range = BitSet.fromRange(start, length, width)
  val testee = range.matches(input)
  val ref = input >= start.U && input < (start + length).U

  assert(testee === ref)
}

class BitSetRangeTest extends AnyFlatSpec with ChiselScalatestTester with Formal {
  val rng = new Random(0x19260817)
  val cases = 128
  "BitSet.fromRange" should "be formally equivalent with bound checks" in {
    for (i <- 1 to cases) {
      val a = rng.nextLong() & Long.MaxValue
      val b = rng.nextLong() & Long.MaxValue
      val start = a.min(b)
      val len = a.max(b) - start + 1
      verify(new BitSetRangeTestModule(start, len, 64), Seq(BoundedCheck(1)))
    }
  }
}
