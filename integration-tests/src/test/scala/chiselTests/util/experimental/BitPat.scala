// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import chisel3._
import chisel3.util.experimental.BitSetRange
import chiseltest._
import chiseltest.formal._
import scala.util.Random
import org.scalatest.flatspec.AnyFlatSpec
import chisel3.util.experimental.BitSet

class BitSetRangeTestModule(start: BigInt, length: BigInt, width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  val range = BitSetRange(start, length, width)
  val testee = range.matches(input)
  val ref = input >= start.U && input < (start + length).U

  assert(testee === ref)
}

class BitSetRangeTest extends AnyFlatSpec with ChiselScalatestTester with Formal {
  val rng = new Random(0x19260817)
  val cases = 128
  "BitSetRange" should "be identical as comparesions" in {
    for(i <- 1 to cases) {
      val a = rng.nextLong() & Long.MaxValue
      val b = rng.nextLong() & Long.MaxValue
      val start = a.min(b)
      val len = a.max(b) - start + 1
      verify(new BitSetRangeTestModule(start, len, 64), Seq(BoundedCheck(1)))
    }
  }

  it should "correctly calculate isAligned" in {
    for(i <- 1 to cases) {
      val start = rng.nextLong() & Long.MaxValue
      val len = ((rng.nextLong() & Long.MaxValue) % 16) + 1

      val range = BitSetRange(start, len, 64)
      require(range.isAligned == (range.terms.size == 1))
    }
  }

  it should "correctly calculate toBitSetRanges" in {
    for(i <- 1 to cases) {
      // Randomly generates 4 patterns with length of 8
      val tmpls = for(j <- 1 to 4) yield {
        val tmpl = for(k <- 1 to 8) yield Seq('?', '0', '1')(rng.between(0, 3))
        "b" + tmpl.mkString
      }

      val set = BitSet.fromString(tmpls.mkString("\n"))

      var lastMax = BigInt(-1)

      var total = BitSet.empty

      for(range <- set.toBitSetRanges) {
        // Ranges must be increasing and non-overlapping (also, not adjacent)
        require(lastMax < range.start)
        lastMax = range.start + range.length
        total = total.union(range)
      }

      require(total.cover(set) && set.cover(total))
    }
  }
}