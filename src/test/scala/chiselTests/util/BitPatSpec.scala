// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.util.BitPat
import _root_.circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BitPatSpec extends AnyFlatSpec with Matchers {
  behavior.of(classOf[BitPat].toString)

  it should "convert a BitPat to readable form" in {
    val testPattern = "0" * 32 + "1" * 32 + "?" * 32 + "?01" * 32
    BitPat("b" + testPattern).toString should be(s"BitPat($testPattern)")
  }

  it should "convert a BitPat to raw form" in {
    val testPattern = "0" * 32 + "1" * 32 + "?" * 32 + "?01" * 32
    BitPat("b" + testPattern).rawString should be(testPattern)
  }

  it should "not fail if BitPat width is 0" in {
    intercept[IllegalArgumentException] { BitPat("b") }
  }

  it should "concat BitPat via ##" in {
    (BitPat.Y(4) ## BitPat.dontCare(3) ## BitPat.N(2)).toString should be(s"BitPat(1111???00)")
  }

  it should "throw when BitPat apply to a Hardware" in {
    intercept[java.lang.IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new chisel3.Module {
        BitPat(chisel3.Reg(chisel3.Bool()))
      })
    }
  }

  it should "index and return new BitPat" in {
    val b = BitPat("b1001???")
    b(0) should be(BitPat.dontCare(1))
    b(6) should be(BitPat.Y())
    b(5) should be(BitPat.N())
  }

  it should "slice and return new BitPat" in {
    val b = BitPat("b1001???")
    b(2, 0) should be(BitPat("b???"))
    b(4, 3) should be(BitPat("b01"))
    b(6, 6) should be(BitPat("b1"))
  }

  it should "parse UInt literals correctly" in {
    BitPat(0.U) should be(new BitPat(0, 1, 1))
    // Note that this parses as 1-bit width, there are other APIs that don't support zero-width UInts correctly
    BitPat(0.U(0.W)) should be(new BitPat(0, 1, 1))
    BitPat(1.U) should be(new BitPat(1, 1, 1))
    BitPat(2.U) should be(new BitPat(2, 3, 2))
    BitPat(0xdeadbeefL.U) should be(new BitPat(BigInt("deadbeef", 16), BigInt("ffffffff", 16), 32))
  }
}
