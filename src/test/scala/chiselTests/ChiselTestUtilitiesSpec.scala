// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import org.scalatest.exceptions.TestFailedException

class ChiselTestUtilitiesSpec extends ChiselFlatSpec {
  // Who tests the testers?
  "assertKnownWidth" should "error when the expected width is wrong" in {
    intercept[TestFailedException] {
      assertKnownWidth(7) {
        Wire(UInt(8.W))
      }
    }
  }

  it should "error when the width is unknown" in {
    intercept[ChiselException] {
      assertKnownWidth(7) {
        Wire(UInt())
      }
    }
  }

  it should "work if the width is correct" in {
    assertKnownWidth(8) {
      Wire(UInt(8.W))
    }
  }

  "assertInferredWidth" should "error if the width is known" in {
    intercept[TestFailedException] {
      assertInferredWidth(8) {
        Wire(UInt(8.W))
      }
    }
  }

  it should "error if the expected width is wrong" in {
    a[TestFailedException] shouldBe thrownBy {
      assertInferredWidth(8) {
        val w = Wire(UInt())
        w := 2.U(2.W)
        w
      }
    }
  }

  it should "pass if the width is correct" in {
    assertInferredWidth(4) {
      val w = Wire(UInt())
      w := 2.U(4.W)
      w
    }
  }
}
