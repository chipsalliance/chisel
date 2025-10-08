// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.reflect.DataMirror
import circt.stage.ChiselStage
import org.scalactic.source.Position
import org.scalatest.Assertions
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

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

class ZeroWidthBundle extends Bundle {
  val x = UInt(0.W)
  val y = UInt()
}
object ZeroWidthBundle {
  def intoWire(): ZeroWidthBundle = {
    val w = Wire(new ZeroWidthBundle)
    w.x := 0.U(0.W)
    w.y := 0.U(0.W)
    w
  }
}

class WidthSpec extends AnyFlatSpec with Matchers {
  "Literals without specified widths" should "get the minimum legal width" in {
    "hdeadbeef".U.getWidth should be(32)
    "h_dead_beef".U.getWidth should be(32)
    "h0a".U.getWidth should be(4)
    "h1a".U.getWidth should be(5)
    "h0".U.getWidth should be(1) // no literal is zero-width unless explicitly requested.
    1.U.getWidth should be(1)
    1.S.getWidth should be(2)
    0.U.getWidth should be(1)
    0.S.getWidth should be(1)
  }
}

trait WidthHelpers extends Assertions {

  def assertKnownWidth(expected: Int, args: Iterable[String] = Nil)(gen: => Data)(implicit pos: Position): Unit = {
    class TestModule extends Module {
      val testPoint = gen
      assert(testPoint.getWidth === expected)
      val out = IO(chiselTypeOf(testPoint))
      // Sanity check that firrtl doesn't change the width
      val zero = 0.U(0.W).asTypeOf(chiselTypeOf(testPoint))
      if (DataMirror.isWire(testPoint)) {
        testPoint := zero
      }
      out := zero
      out := testPoint
    }
    val verilog = ChiselStage.emitSystemVerilog(new TestModule, args.toArray, Array("-disable-all-randomization"))
    expected match {
      case 0 => assert(!verilog.contains("out"))
      case 1 =>
        assert(verilog.contains(s"out"))
        assert(!verilog.contains(s"0] out"))
      case _ => assert(verilog.contains(s"[${expected - 1}:0] out"))
    }
  }

  def assertInferredWidth(expected: Int, args: Iterable[String] = Nil)(gen: => Data)(implicit pos: Position): Unit = {
    class TestModule extends Module {
      val testPoint = gen
      assert(!testPoint.isWidthKnown, s"Asserting that width should be inferred yet width is known to Chisel!")
      // Sanity check that firrtl doesn't change the width
      val widthcheck = Wire(chiselTypeOf(testPoint))
      dontTouch(widthcheck)
      val zero = 0.U(0.W).asTypeOf(chiselTypeOf(testPoint))
      if (DataMirror.isWire(testPoint)) {
        testPoint := zero
      }
      widthcheck := zero
      widthcheck := testPoint
    }
    val verilog =
      ChiselStage.emitSystemVerilog(new TestModule, args.toArray, Array("-disable-all-randomization"))
    expected match {
      case 0 => assert(!verilog.contains("widthcheck"))
      case 1 =>
        assert(verilog.contains(s"widthcheck"))
        assert(!verilog.contains(s"0] widthcheck"))
      case _ => assert(verilog.contains(s"[${expected - 1}:0] widthcheck"))
    }
  }

}

abstract class WireRegWidthSpecImpl extends AnyFlatSpec with Matchers with WidthHelpers {
  def name:                     String
  def builder[T <: Data](x: T): T

  behavior.of(name)

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

  it should "set the width to zero if the template type is set to zero-width" in {
    assertKnownWidth(0) {
      builder(UInt(0.W))
    }
    assertKnownWidth(0) {
      val w = builder(new ZeroWidthBundle)
      w := ZeroWidthBundle.intoWire()
      w.x
    }
    assertKnownWidth(0) {
      val x = builder(Vec(1, UInt(0.W)))
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

  it should "infer width as zero if the template type has no width and is initialized to zero-width literal" in {
    assertInferredWidth(0) {
      val w = builder(UInt())
      w := 0.U(0.W)
      w
    }
    assertInferredWidth(0) {
      val w = builder(new ZeroWidthBundle)
      w := ZeroWidthBundle.intoWire()
      w.y
    }
    assertInferredWidth(0) {
      val w = builder(Vec(1, UInt()))
      w(0) := 0.U(0.W)
      w(0)
    }
  }
}

class WireWidthSpec extends WireRegWidthSpecImpl {
  def name:                     String = "Wire"
  def builder[T <: Data](x: T): T = Wire(x)
}
class RegWidthSpec extends WireRegWidthSpecImpl {
  def name:                     String = "Reg"
  def builder[T <: Data](x: T): T = Reg(x)
}

abstract class WireDefaultRegInitSpecImpl extends AnyFlatSpec with Matchers with WidthHelpers {
  def name:                            String
  def builder1[T <: Data](x: T):       T
  def builder2[T <: Data](x: T, y: T): T

  behavior.of(s"$name (Single Argument)")

  it should "set width if passed a literal with forced width" in {
    assertKnownWidth(4) {
      builder1(3.U(4.W))
    }
    assertKnownWidth(0) {
      builder1(0.U(0.W))
    }
  }

  it should "NOT set width if passed a literal without a forced width" in {
    assertInferredWidth(4) {
      val w = builder1(3.U)
      w := 3.U(4.W)
      w
    }

    assertInferredWidth(1) {
      val w = builder1(0.U)
      w := 0.U(0.W)
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

  behavior.of(s"$name (Double Argument)")

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
      () => 0.U,
      () => 0.U(2.W),
      () => WireDefault(0.U),
      () => WireDefault(0.U(2.W))
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
  def name:                            String = "WireDefault"
  def builder1[T <: Data](x: T):       T = WireDefault(x)
  def builder2[T <: Data](x: T, y: T): T = WireDefault(x, y)
}

class RegInitWidthSpec extends WireDefaultRegInitSpecImpl {
  def name:                            String = "RegInit"
  def builder1[T <: Data](x: T):       T = RegInit(x)
  def builder2[T <: Data](x: T, y: T): T = RegInit(x, y)
}
