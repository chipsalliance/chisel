// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec

class GrayCodeTests extends AnyFlatSpec with ChiselScalatestTester with Formal {
  behavior.of("GrayCode")

  val Widths = Seq(1, 2, 3, 5, 8, 17, 65)
  Widths.foreach { w =>
    it should s"maintain identity (width=$w)" in {
      verify(new GrayCodeIdentityCheck(w), Seq(BoundedCheck(1)))
    }

    it should s"ensure hamming distance of one (width=$w)" in {
      verify(new GrayCodeHammingCheck(w), Seq(BoundedCheck(1)))
    }
  }
}

/** Checks that when we go from binary -> gray -> binary the result is always the same as the input. */
private class GrayCodeIdentityCheck(width: Int) extends Module {
  val in = IO(Input(UInt(width.W)))
  val gray = BinaryToGray(in)
  val out = GrayToBinary(gray)
  assert(in === out, "%b -> %b -> %b", in, gray, out)
}

/** Checks that if we increment the binary number, the gray code equivalent only changes by one bit. */
private class GrayCodeHammingCheck(width: Int) extends Module {
  val a = IO(Input(UInt(width.W)))
  val b = a + 1.U
  val aGray = BinaryToGray(a)
  val bGray = BinaryToGray(b)
  val hamming = PopCount(aGray ^ bGray)
  assert(hamming === 1.U, "%b ^ %b = %b", aGray, bGray, hamming)
}
