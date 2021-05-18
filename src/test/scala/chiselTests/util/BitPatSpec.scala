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

  it should "not fail if BitPat width is 0" in {
    intercept[IllegalArgumentException]{BitPat("b")}
  }
}
