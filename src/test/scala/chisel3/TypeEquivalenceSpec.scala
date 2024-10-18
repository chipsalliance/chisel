// SPDX-License-Identifier: Apache-2.0

package chisel3

import circt.stage.ChiselStage.emitCHIRRTL
import chisel3.experimental.Analog
import chisel3.probe.{Probe, RWProbe}
import chisel3.layer.Layer

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers._

object TypeEquivalenceSpec {
  class Foo(hasC: Boolean = true) extends Bundle {
    val a = UInt(8.W)
    val b = UInt(8.W)
    val c = if (hasC) Some(UInt(8.W)) else None
  }

  class FooParent(hasC: Boolean) extends Bundle {
    val foo = new Foo(hasC)
  }

  class FooGrandparent(hasC: Boolean) extends Bundle {
    val bar = new FooParent(hasC)
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

  class BundleWithProbe(useProbe: Boolean) extends Bundle {
    val logic = Bool()
    val maybeProbe = if (useProbe) Probe(Bool()) else Bool()
  }

  object Red extends Layer(layer.Convention.Bind) {
    override def toString = "Red"
  }
  object Green extends Layer(layer.Convention.Bind) {
    override def toString = "Green"
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
    x.findFirstTypeMismatch(y, true, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true, true) should be(
      Some(": Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )
    x.findFirstTypeMismatch(z, true, false, true) should be(None)
    x.findFirstTypeMismatch(zz, true, true, true) should be(
      Some(": Left (UInt<8>) and Right (UInt) have different widths.")
    )
    x.findFirstTypeMismatch(zz, true, false, true) should be(None)
  }

  it should "support comparing SInts" in {
    val x = SInt(8.W)
    val y = SInt(8.W)
    val z = SInt(4.W)
    val zz = SInt()
    x.findFirstTypeMismatch(y, true, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true, true) should be(
      Some(": Left (SInt<8>) and Right (SInt<4>) have different widths.")
    )
    x.findFirstTypeMismatch(z, true, false, true) should be(None)
    x.findFirstTypeMismatch(zz, true, true, true) should be(
      Some(": Left (SInt<8>) and Right (SInt) have different widths.")
    )
    x.findFirstTypeMismatch(zz, true, false, true) should be(None)
  }

  it should "catch comparing SInts and UInts" in {
    val x = UInt(8.W)
    val y = SInt(8.W)
    x.findFirstTypeMismatch(y, true, true, true) should be(
      Some(": Left (UInt<8>) and Right (SInt<8>) have different types.")
    )
  }

  it should "support equivalent Bundles" in {
    val x = new Foo(true)
    val y = new Foo(true)
    x.findFirstTypeMismatch(y, true, true, true) should be(None)
  }

  it should "support structurally equivalent Bundles" in {
    val x = new Foo(false)
    val y = new Bar
    x.findFirstTypeMismatch(y, true, true, true) should be(Some(": Left (Foo) and Right (Bar) have different types."))
    x.findFirstTypeMismatch(y, false, true, true) should be(None)
  }

  it should "support Vecs" in {
    val x = Vec(2, UInt(8.W))
    val y = Vec(2, UInt(8.W))
    val z = Vec(3, UInt(8.W))
    x.findFirstTypeMismatch(y, true, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true, true) should be(
      Some(": Left (size 2) and Right (size 3) have different lengths.")
    )
  }

  it should "support nested width mismatches" in {
    val x = new Bar(8)
    val y = new Bar(4)
    x.findFirstTypeMismatch(y, true, false, true) should be(None)
    x.findFirstTypeMismatch(y, true, true, true) should be(
      Some(".b: Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )

    val a = Vec(4, new Bar(8))
    val b = Vec(4, new Bar(4))
    a.findFirstTypeMismatch(b, true, false, true) should be(None)
    a.findFirstTypeMismatch(b, true, true, true) should be(
      Some("[_].b: Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )

    val c = new Bundle {
      val foo = new Bar(8)
    }
    val d = new Bundle {
      val foo = new Bar(4)
    }
    c.findFirstTypeMismatch(d, false, false, true) should be(None)
    c.findFirstTypeMismatch(d, false, true, true) should be(
      Some(".foo.b: Left (UInt<8>) and Right (UInt<4>) have different widths.")
    )
  }

  it should "support equivalent ChiselEnums" in {
    val x = Fizz()
    val y = Fizz()
    val z = Buzz()
    x.findFirstTypeMismatch(y, true, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true, true) should be(Some(": Left (Fizz) and Right (Buzz) have different types."))
    // TODO should there be some form of structural typing for ChiselEnums?
    x.findFirstTypeMismatch(z, false, true, true) should be(
      Some(": Left (Fizz) and Right (Buzz) have different types.")
    )
  }

  it should "support comparing Analogs" in {
    val x = Analog(8.W)
    val y = Analog(8.W)
    val z = Analog(4.W)
    x.findFirstTypeMismatch(y, true, true, true) should be(None)
    x.findFirstTypeMismatch(z, true, true, true) should be(
      Some(": Left (Analog<8>) and Right (Analog<4>) have different widths.")
    )
    x.findFirstTypeMismatch(z, true, false, true) should be(None)
  }

  it should "support DontCare" in {
    DontCare.findFirstTypeMismatch(DontCare, true, true, true) should be(None)
  }

  it should "support AsyncReset" in {
    AsyncReset().findFirstTypeMismatch(AsyncReset(), true, true, true) should be(None)
    AsyncReset().findFirstTypeMismatch(Bool(), true, true, true) should be(
      Some(": Left (AsyncReset) and Right (Bool) have different types.")
    )
    AsyncReset().findFirstTypeMismatch(Clock(), true, true, true) should be(
      Some(": Left (AsyncReset) and Right (Clock) have different types.")
    )
    AsyncReset().findFirstTypeMismatch(Reset(), true, true, true) should be(
      Some(": Left (AsyncReset) and Right (Reset) have different types.")
    )
  }

  it should "support Clock" in {
    Clock().findFirstTypeMismatch(Clock(), true, true, true) should be(None)
  }

  it should "support abstract Reset" in {
    Reset().findFirstTypeMismatch(Reset(), true, true, true) should be(None)
  }

  it should "support Probe" in {
    Probe(Bool()).findFirstTypeMismatch(Probe(Bool()), true, true, true) should be(None)
  }

  it should "support RWProbe" in {
    RWProbe(Bool()).findFirstTypeMismatch(RWProbe(Bool()), true, true, true) should be(None)
  }

  it should "detect differences between Probe and Not-Probe" in {
    Probe(Bool()).findFirstTypeMismatch(Bool(), true, true, true) should be(
      Some(
<<<<<<< HEAD
        ": Left (Bool with probeInfo: Some(writeable=false)) and Right (Bool with probeInfo: None) have different probeInfo."
=======
        ": Left (Probe<Bool> with probeInfo: Some(writeable=false, color=None)) and Right (Bool with probeInfo: None) have different probeInfo."
>>>>>>> 8c718b2b6 (Add Probes to .toString Data methods (#4478))
      )
    )
  }

  it should "work for Bundles with Probes" in {
    new BundleWithProbe(true).findFirstTypeMismatch(new BundleWithProbe(true), true, true, true) should be(None)
  }

  it should "detect differences between Probe and Not-Probe within a Bundle" in {
    new BundleWithProbe(true).findFirstTypeMismatch(new BundleWithProbe(false), true, true, true) should be(
      Some(
<<<<<<< HEAD
        ".maybeProbe: Left (Bool with probeInfo: Some(writeable=false)) and Right (Bool with probeInfo: None) have different probeInfo."
=======
        ".maybeProbe: Left (Probe<Bool> with probeInfo: Some(writeable=false, color=None)) and Right (Bool with probeInfo: None) have different probeInfo."
>>>>>>> 8c718b2b6 (Add Probes to .toString Data methods (#4478))
      )
    )
  }

  it should "detect differences between probe types" in {
    RWProbe(Bool()).findFirstTypeMismatch(Probe(Bool()), true, true, true) should be(
      Some(
<<<<<<< HEAD
        ": Left (Bool with probeInfo: Some(writeable=true)) and Right (Bool with probeInfo: Some(writeable=false)) have different probeInfo."
=======
        ": Left (RWProbe<Bool> with probeInfo: Some(writeable=true, color=None)) and Right (Probe<Bool> with probeInfo: Some(writeable=false, color=None)) have different probeInfo."
>>>>>>> 8c718b2b6 (Add Probes to .toString Data methods (#4478))
      )
    )
  }

  it should "detect differences through probes" in {
    Probe(Bool()).findFirstTypeMismatch(Probe(Clock()), true, true, true) should be(
      Some(": Left (Probe<Bool>) and Right (Probe<Clock>) have different types.")
    )
  }

<<<<<<< HEAD
  it should "detect differences in probe within a Vector" in {
    Vec(3, Probe(Bool())).findFirstTypeMismatch(Vec(3, Bool()), true, true, true) should be(
      Some(
        "[_]: Left (Bool with probeInfo: Some(writeable=false)) and Right (Bool with probeInfo: None) have different probeInfo."
=======
  it should "detect differences in presence of probe colors" in {
    Probe(Bool()).findFirstTypeMismatch(Probe(Bool(), Green), true, true, true) should be(
      Some(
        ": Left (Probe<Bool> with probeInfo: Some(writeable=false, color=None)) and Right (Probe[Green]<Bool> with probeInfo: Some(writeable=false, color=Some(Green))) have different probeInfo."
      )
    )
  }

  it should "detect differences in probe colors" in {
    Probe(Bool(), Red).findFirstTypeMismatch(Probe(Bool(), Green), true, true, true) should be(
      Some(
        ": Left (Probe[Red]<Bool> with probeInfo: Some(writeable=false, color=Some(Red))) and Right (Probe[Green]<Bool> with probeInfo: Some(writeable=false, color=Some(Green))) have different probeInfo."
      )
    )
  }

  it should "work with probes within a Bundle" in {
    new BundleWithAColor(Some(Red)).findFirstTypeMismatch(new BundleWithAColor(Some(Red)), true, true, true) should be(
      None
    )
  }

  it should "detect differences in probe colors within a Bundle" in {
    new BundleWithAColor(Some(Red)).findFirstTypeMismatch(
      new BundleWithAColor(Some(Green)),
      true,
      true,
      true
    ) should be(
      Some(
        ".probe: Left (Probe[Red]<Bool> with probeInfo: Some(writeable=false, color=Some(Red))) and Right (Probe[Green]<Bool> with probeInfo: Some(writeable=false, color=Some(Green))) have different probeInfo."
      )
    )
  }

  it should "detect differences in probe color presence within a Bundle" in {
    new BundleWithAColor(Some(Red)).findFirstTypeMismatch(new BundleWithAColor(None), true, true, true) should be(
      Some(
        ".probe: Left (Probe[Red]<Bool> with probeInfo: Some(writeable=false, color=Some(Red))) and Right (Probe<Bool> with probeInfo: Some(writeable=false, color=None)) have different probeInfo."
      )
    )
  }

  it should "detect differences in probe within a Vector" in {
    Vec(3, Probe(Bool())).findFirstTypeMismatch(Vec(3, Bool()), true, true, true) should be(
      Some(
        "[_]: Left (Probe<Bool> with probeInfo: Some(writeable=false, color=None)) and Right (Bool with probeInfo: None) have different probeInfo."
>>>>>>> 8c718b2b6 (Add Probes to .toString Data methods (#4478))
      )
    )
  }

  behavior.of("Data.requireTypeEquivalent")

  it should "have a good user message if it fails for wrong scala type" in {
    val result = the[IllegalArgumentException] thrownBy {
      Bool().requireTypeEquivalent(UInt(1.W), "This test should fail, because: ")
    }
    result.getMessage should include("This test should fail, because: Bool is not typeEquivalent to UInt<1>")
  }

  it should "have a good user message if it fails for mismatched bundles" in {
    val result = the[IllegalArgumentException] thrownBy {
      (new FooGrandparent(true)).requireTypeEquivalent(new FooGrandparent(false), "This test should fail, because: ")
    }
    result.getMessage should include("This test should fail, because:")
    // Also says 'FooGrandparent$1 is not typeEquivalent to FooGrandparent$1' which isn't terribly interesting to match on
    result.getMessage should include(".bar.foo.c: Dangling field on Left")
  }

}
