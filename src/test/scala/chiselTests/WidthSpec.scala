// See LICENSE for license details.

package chiselTests

import chisel3._

class SimpleBundle extends Bundle {
  val x = UInt(4.W)
  val y = UInt()
}
object SimpleBundle {
  def intoWire(): SimpleBundle = {
    val w = Wire(new SimpleBundle)
    w.x := 0.U(4.W)
    w.y := 0.U(4.W)
    w
  }
}

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
}

abstract class WireRegWidthSpecImpl extends ChiselFlatSpec {
  def name: String
  def builder[T <: Data](x: T): T

  behavior of name

  it should "set the width if the template type has a set width" in {
    assertKnownWidth(4) {
      builder(UInt(4.W))
    }
    assertKnownWidth(4) {
      val w = builder(new SimpleBundle)
      w := SimpleBundle.intoWire()
      w.x
    }
    assertKnownWidth(4) {
      val x = builder(Vec(1, UInt(4.W)))
      x(0)
    }
  }

  it should "infer the width if the template type has no width" in {
    assertInferredWidth(4) {
      val w = builder(UInt())
      w := 0.U(4.W)
      w
    }
    assertInferredWidth(4) {
      val w = builder(new SimpleBundle)
      w := SimpleBundle.intoWire()
      w.y
    }
    assertInferredWidth(4) {
      val w = builder(Vec(1, UInt()))
      w(0) := 0.U(4.W)
      w(0)
    }
  }
}

class WireWidthSpec extends WireRegWidthSpecImpl {
  def name: String = "Wire"
  def builder[T <: Data](x: T): T = Wire(x)
}
class RegWidthSpec extends WireRegWidthSpecImpl {
  def name: String = "Reg"
  def builder[T <: Data](x: T): T = Reg(x)
}

abstract class WireDefaultRegInitSpecImpl extends ChiselFlatSpec {
  def name: String
  def builder1[T <: Data](x: T): T
  def builder2[T <: Data](x: T, y: T): T

  behavior of s"$name (Single Argument)"

  it should "set width if passed a literal with forced width" in {
    assertKnownWidth(4) {
      builder1(3.U(4.W))
    }
  }

  it should "NOT set width if passed a literal without a forced width" in {
    assertInferredWidth(4) {
      val w = builder1(3.U)
      w := 3.U(4.W)
      w
    }
  }

  it should "NOT set width if passed a non-literal" in {
    assertInferredWidth(4) {
      val w = WireDefault(3.U(4.W))
      builder1(w)
    }
  }

  it should "copy the widths of aggregates" in {
    assertInferredWidth(4) {
      val w = builder1(SimpleBundle.intoWire())
      w.y
    }
    assertKnownWidth(4) {
      val w = builder1(SimpleBundle.intoWire())
      w.x
    }
    assertInferredWidth(4) {
      val x = Wire(Vec(1, UInt()))
      x(0) := 0.U(4.W)
      val w = builder1(x)
      w(0)
    }
    assertKnownWidth(4) {
      val x = Wire(Vec(1, UInt(4.W)))
      x(0) := 0.U(4.W)
      val w = builder1(x)
      w(0)
    }
  }

  behavior of s"$name (Double Argument)"

  it should "set the width if the template type has a set width" in {
    assertKnownWidth(4) {
      WireDefault(UInt(4.W), 0.U)
    }
    assertKnownWidth(4) {
      WireDefault(UInt(4.W), 0.U(2.W))
    }
    assertKnownWidth(4) {
      val w = WireDefault(new SimpleBundle, SimpleBundle.intoWire())
      w.x
    }
    assertKnownWidth(4) {
      val x = Wire(Vec(1, UInt()))
      x(0) := 0.U(4.W)
      val w = WireDefault(Vec(1, UInt(4.W)), x)
      w(0)
    }
  }

  it should "infer the width if the template type has no width" in {
    val templates = Seq(
      () => 0.U, () => 0.U(2.W), () => WireDefault(0.U), () => WireDefault(0.U(2.W))
    )
    for (gen <- templates) {
      assertInferredWidth(4) {
        val w = WireDefault(UInt(), gen())
        w := 0.U(4.W)
        w
      }
    }
    assertInferredWidth(4) {
      val w = WireDefault(new SimpleBundle, SimpleBundle.intoWire())
      w.y
    }
    assertInferredWidth(4) {
      val x = Wire(Vec(1, UInt()))
      x(0) := 0.U(4.W)
      val w = WireDefault(Vec(1, UInt()), x)
      w(0)
    }
  }
}

class WireDefaultWidthSpec extends WireDefaultRegInitSpecImpl {
  def name: String = "WireDefault"
  def builder1[T <: Data](x: T): T = WireDefault(x)
  def builder2[T <: Data](x: T, y: T): T = WireDefault(x, y)
}

class RegInitWidthSpec extends WireDefaultRegInitSpecImpl {
  def name: String = "RegInit"
  def builder1[T <: Data](x: T): T = RegInit(x)
  def builder2[T <: Data](x: T, y: T): T = RegInit(x, y)
}

