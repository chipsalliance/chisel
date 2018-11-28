// See LICENSE for license details.

package chiselTests

import chisel3._

class WidthSpec extends ChiselFlatSpec {
  "Literals without specified widths" should "get the minimum legal width" in {
    "hdeadbeef".U.getWidth should be (32)
    "h_dead_beef".U.getWidth should be (32)
    "h0a".U.getWidth should be (4)
    "h1a".U.getWidth should be (5)
    "h0".U.getWidth should be (1)
    1.U.getWidth should be (1)
    1.S.getWidth should be (2)
  }

  behavior of "WireInit (Single Argument)"

  it should "set width if passed a literal with forced width" in {
    assertKnownWidth(4) {
      WireInit(3.U(4.W))
    }
  }

  it should "NOT set width if passed a literal without a forced width" in {
    assertInferredWidth(4) {
      val w = WireInit(3.U)
      w := 3.U(4.W)
      w
    }
  }

  it should "NOT set width if passed a non-literal" in {
    assertInferredWidth(4) {
      val w = WireInit(3.U(4.W))
      WireInit(w)
    }
  }

  behavior of "WireInit (Double Argument)"

  it should "set the width if the template type has a set width" in {
    assertKnownWidth(4) {
      WireInit(UInt(4.W), 0.U)
    }
    assertKnownWidth(4) {
      WireInit(UInt(4.W), 0.U(2.W))
    }
  }

  it should "infer the width if the template type has no width" in {
    val templates = Seq(
      () => 0.U, () => 0.U(2.W), () => WireInit(0.U), () => WireInit(0.U(2.W))
    )
    for (gen <- templates) {
      assertInferredWidth(4) {
        val w = WireInit(UInt(), gen())
        w := 0.U(4.W)
        w
      }
    }
  }

  behavior of "RegInit (Single Argument)"

  it should "set width if passed a literal with forced width" in {
    assertKnownWidth(4) {
      RegInit(3.U(4.W))
    }
  }

  it should "NOT set width if passed a literal without a forced width" in {
    assertInferredWidth(4) {
      val w = RegInit(3.U)
      w := 3.U(4.W)
      w
    }
  }

  it should "NOT set width if passed a non-literal" in {
    assertInferredWidth(4) {
      val w = WireInit(3.U(4.W))
      RegInit(w)
    }
  }

  behavior of "RegInit (Double Argument)"

  it should "set the width if the template type has a set width" in {
    assertKnownWidth(4) {
      RegInit(UInt(4.W), 0.U)
    }
    assertKnownWidth(4) {
      RegInit(UInt(4.W), 0.U(2.W))
    }
  }

  it should "infer the width if the template type has no width" in {
    val templates = Seq(
      () => 0.U, () => 0.U(2.W), () => WireInit(0.U), () => WireInit(0.U(2.W))
    )
    for (gen <- templates) {
      assertInferredWidth(4) {
        val w = RegInit(UInt(), gen())
        w := 0.U(4.W)
        w
      }
    }
  }
}

