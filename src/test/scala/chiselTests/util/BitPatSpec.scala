// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3.util.BitPat
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


class BitPatSpec extends AnyFlatSpec with Matchers {
  behavior of classOf[BitPat].toString

  it should "convert a BitPat to readable form" in {
    val testPattern = "0" * 32 + "1" * 32 + "?" * 32 + "?01" * 32
    BitPat("b" + testPattern).toString should be (s"BitPat($testPattern)")
  }

  it should "convert a BitPat to raw form" in {
    val testPattern = "0" * 32 + "1" * 32 + "?" * 32 + "?01" * 32
    BitPat("b" + testPattern).rawString should be(testPattern)
  }

  it should "not fail if BitPat width is 0" in {
    intercept[IllegalArgumentException]{BitPat("b")}
  }

  it should "contact BitPat via ##" in {
    (BitPat.Y(4) ## BitPat.dontCare(3) ## BitPat.N(2)).toString should be (s"BitPat(1111???00)")
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

  it should "have a reflective overlap relation" in {
    val a = BitPat("b1001???")
    val b = BitPat("b10?1???")
    val c = BitPat("b10?0???")
    a.overlap(a) should be(true)
    a.overlap(b) should be(true)
    a.overlap(c) should be(false)
    b.overlap(a) should be(true)
    c.overlap(a) should be(false)
  }

  it should "have a anti-reflective contain relation" in {
    val a = BitPat("b1001???")
    val b = BitPat("b10?1???")
    val c = BitPat("b10?0???")
    b.contain(b) should be(true)
    b.contain(a) should be(true)
    b.contain(c) should be(false)
    a.contain(b) should be(false)
  }

  it should "be able to calculate intersection with another BitPat" in {
    val a = BitPat("b1??")
    val b = BitPat("b0??")
    val c = BitPat("b??1")
    a.intersect(a) should be(Set(a))
    a.intersect(b) should be(Set())
    a.intersect(c) should be(Set(BitPat("b1?1")))
  }

  it should "be able to subtract from another BitPat" in {
    val a = BitPat("b??")
    val b = BitPat("b?0")
    val c = BitPat("b00")
    a.subtract(a) should be(Set())
    a.subtract(b) should be(Set(BitPat("b?1")))
    a.subtract(c) should be(Set(BitPat("b01"), BitPat("b1?")))
  }
}
