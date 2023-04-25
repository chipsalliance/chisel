// SPDX-License-Identifier: Apache-2.0

package chisel3

import circt.stage.ChiselStage.emitCHIRRTL
import chisel3.experimental.Analog

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers._

object TypeEquivalenceSpec {
  class Foo(hasC: Boolean = true) extends Bundle {
    val a = UInt(8.W)
    val b = UInt(8.W)
    val c = if (hasC) Some(UInt(8.W)) else None
  }

  class Bar(bWidth: Int = 8) extends Bundle {
    val a = UInt(8.W)
    val b = UInt(bWidth.W)
  }

  object Fizz extends ChiselEnum {
    val e0, e1 = Value
    val e4 = Value(4.U)
  }

  object Buzz extends ChiselEnum {
    val e0, e1 = Value
    val e4 = Value(4.U)
  }
}

import TypeEquivalenceSpec._

class TypeEquivalenceSpec extends AnyFlatSpec {
  behavior.of("Data.typeEquivalent")

  it should "support comparing UInts" in {
    val x = UInt(8.W)
    val y = UInt(8.W)
    val z = UInt(4.W)
    assert(x.typeEquivalent(y))
    assert(!x.typeEquivalent(z))
    assert(!x.typeEquivalent(UInt()))
    assert(UInt().typeEquivalent(UInt()))
  }

  it should "support comparing SInts" in {
    val x = SInt(8.W)
    val y = SInt(8.W)
    val z = SInt(4.W)
    assert(x.typeEquivalent(y))
    assert(!x.typeEquivalent(z))
    assert(!x.typeEquivalent(SInt()))
    assert(SInt().typeEquivalent(SInt()))
  }

  it should "catch comparing SInts and UInts" in {
    val x = UInt(8.W)
    val y = SInt(8.W)
    assert(!x.typeEquivalent(y))
  }

  it should "support equivalent Bundles" in {
    val x = new Foo(true)
    val y = new Foo(true)
    assert(x.typeEquivalent(y))
  }

  it should "reject structurally equivalent Bundles" in {
    val x = new Foo(false)
    val y = new Bar
    assert(!x.typeEquivalent(y))
  }

  it should "support Vecs" in {
    val x = Vec(2, UInt(8.W))
    val y = Vec(2, UInt(8.W))
    val z = Vec(3, UInt(8.W))
    assert(x.typeEquivalent(y))
    assert(!x.typeEquivalent(z))
  }

  it should "reject nested width mismatches" in {
    val x = new Bar(8)
    val y = new Bar(4)
    assert(!x.typeEquivalent(y))

    val a = Vec(4, new Bar(8))
    val b = Vec(4, new Bar(4))
    assert(!a.typeEquivalent(b))

    val c = new Bundle {
      val foo = new Bar(8)
    }
    val d = new Bundle {
      val foo = new Bar(4)
    }
    assert(!c.typeEquivalent(d))
  }

  it should "support equivalent ChiselEnums" in {
    val x = Fizz()
    val y = Fizz()
    val z = Buzz()
    assert(x.typeEquivalent(y))
    assert(!x.typeEquivalent(z))
  }

  it should "support comparing Analogs" in {
    val x = Analog(8.W)
    val y = Analog(8.W)
    val z = Analog(4.W)
    assert(x.typeEquivalent(y))
    assert(!x.typeEquivalent(z))
  }

  it should "support DontCare" in {
    assert(DontCare.typeEquivalent(DontCare))
    assert(!DontCare.typeEquivalent(UInt()))
  }

  it should "support AsyncReset" in {
    assert(AsyncReset().typeEquivalent(AsyncReset()))
    assert(!AsyncReset().typeEquivalent(Bool()))
    assert(!AsyncReset().typeEquivalent(Clock()))
    assert(!AsyncReset().typeEquivalent(Reset()))
  }

  it should "support Clock" in {
    assert(Clock().typeEquivalent(Clock()))
  }

  it should "support abstract Reset" in {
    assert(Reset().typeEquivalent(Reset()))
  }

  behavior.of("Data.findFirstTypeMismatch")

  it should "support comparing UInts" in {
    val x = UInt(8.W)
    val y = UInt(8.W)
    val z = UInt(4.W)
    val zz = UInt()
    x.findFirstTypeMismatch(y, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true) should be(
      Some(": Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )
    x.findFirstTypeMismatch(z, true, false) should be(None)
    x.findFirstTypeMismatch(zz, true, true) should be(Some(": Left (UInt<8>) and Right (UInt) have different widths."))
    x.findFirstTypeMismatch(zz, true, false) should be(None)
  }

  it should "support comparing SInts" in {
    val x = SInt(8.W)
    val y = SInt(8.W)
    val z = SInt(4.W)
    val zz = SInt()
    x.findFirstTypeMismatch(y, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true) should be(
      Some(": Left (SInt<8>) and Right (SInt<4>) have different widths.")
    )
    x.findFirstTypeMismatch(z, true, false) should be(None)
    x.findFirstTypeMismatch(zz, true, true) should be(Some(": Left (SInt<8>) and Right (SInt) have different widths."))
    x.findFirstTypeMismatch(zz, true, false) should be(None)
  }

  it should "catch comparing SInts and UInts" in {
    val x = UInt(8.W)
    val y = SInt(8.W)
    x.findFirstTypeMismatch(y, true, true) should be(Some(": Left (UInt<8>) and Right (SInt<8>) have different types."))
  }

  it should "support equivalent Bundles" in {
    val x = new Foo(true)
    val y = new Foo(true)
    x.findFirstTypeMismatch(y, true, true) should be(None)
  }

  it should "support structurally equivalent Bundles" in {
    val x = new Foo(false)
    val y = new Bar
    x.findFirstTypeMismatch(y, true, true) should be(Some(": Left (Foo) and Right (Bar) have different types."))
    x.findFirstTypeMismatch(y, false, true) should be(None)
  }

  it should "support Vecs" in {
    val x = Vec(2, UInt(8.W))
    val y = Vec(2, UInt(8.W))
    val z = Vec(3, UInt(8.W))
    x.findFirstTypeMismatch(y, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true) should be(Some(": Left (size 2) and Right (size 3) have different lengths."))
  }

  it should "support nested width mismatches" in {
    val x = new Bar(8)
    val y = new Bar(4)
    x.findFirstTypeMismatch(y, true, false) should be(None)
    x.findFirstTypeMismatch(y, true, true) should be(
      Some(".b: Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )

    val a = Vec(4, new Bar(8))
    val b = Vec(4, new Bar(4))
    a.findFirstTypeMismatch(b, true, false) should be(None)
    a.findFirstTypeMismatch(b, true, true) should be(
      Some("[_].b: Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )

    val c = new Bundle {
      val foo = new Bar(8)
    }
    val d = new Bundle {
      val foo = new Bar(4)
    }
    c.findFirstTypeMismatch(d, false, false) should be(None)
    c.findFirstTypeMismatch(d, false, true) should be(
      Some(".foo.b: Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )
  }

  it should "support equivalent ChiselEnums" in {
    val x = Fizz()
    val y = Fizz()
    val z = Buzz()
    x.findFirstTypeMismatch(y, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true) should be(Some(": Left (Fizz) and Right (Buzz) have different types."))
    // TODO should there be some form of structural typing for ChiselEnums?
    x.findFirstTypeMismatch(z, false, true) should be(Some(": Left (Fizz) and Right (Buzz) have different types."))
  }

  it should "support comparing Analogs" in {
    val x = Analog(8.W)
    val y = Analog(8.W)
    val z = Analog(4.W)
    x.findFirstTypeMismatch(y, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true) should be(
      Some(": Left (Analog<8>) and Right (Analog<4>) have different widths.")
    )
    x.findFirstTypeMismatch(z, true, false) should be(None)
  }

  it should "support DontCare" in {
    DontCare.findFirstTypeMismatch(DontCare, true, true) should be(None)
  }

  it should "support AsyncReset" in {
    AsyncReset().findFirstTypeMismatch(AsyncReset(), true, true) should be(None)
    AsyncReset().findFirstTypeMismatch(Bool(), true, true) should be(
      Some(": Left (AsyncReset) and Right (Bool) have different types.")
    )
    AsyncReset().findFirstTypeMismatch(Clock(), true, true) should be(
      Some(": Left (AsyncReset) and Right (Clock) have different types.")
    )
    AsyncReset().findFirstTypeMismatch(Reset(), true, true) should be(
      Some(": Left (AsyncReset) and Right (Reset) have different types.")
    )
  }

  it should "support Clock" in {
    Clock().findFirstTypeMismatch(Clock(), true, true) should be(None)
  }

  it should "support abstract Reset" in {
    Reset().findFirstTypeMismatch(Reset(), true, true) should be(None)
  }
}
