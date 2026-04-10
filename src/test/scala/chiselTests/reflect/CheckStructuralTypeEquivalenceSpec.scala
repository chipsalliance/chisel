// SPDX-License-Identifier: Apache-2.0

package chiselTests.reflect

import chisel3._
import chisel3.reflect.DataMirror
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CheckStructuralTypeEquivalenceSpec extends AnyFlatSpec with Matchers {

  behavior.of("DataMirror.checkStructuralTypeEquivalence")

  // -- Helper Bundles --

  class BundleA extends Bundle {
    val x = UInt(8.W)
    val y = SInt(16.W)
  }

  // Different class, same fields as BundleA
  class BundleA2 extends Bundle {
    val x = UInt(8.W)
    val y = SInt(16.W)
  }

  class BundleB extends Bundle {
    val x = UInt(8.W)
    val z = SInt(16.W) // different field name
  }

  class BundleC extends Bundle {
    val x = UInt(8.W)
    val y = UInt(16.W) // different field type (UInt vs SInt)
  }

  class BundleD extends Bundle {
    val x = UInt(8.W)
    val y = SInt(32.W) // different width
  }

  class NestedBundle extends Bundle {
    val inner = new BundleA
    val flag  = Bool()
  }

  // Different class, same nested structure
  class NestedBundle2 extends Bundle {
    val inner = new BundleA2
    val flag  = Bool()
  }

  class VecBundle extends Bundle {
    val data = Vec(4, UInt(8.W))
  }

  class VecBundle2 extends Bundle {
    val data = Vec(4, UInt(8.W))
  }

  class VecBundleDiffLen extends Bundle {
    val data = Vec(3, UInt(8.W))
  }

  class EmptyBundle extends Bundle

  class EmptyBundle2 extends Bundle

  // -- Tests --

  it should "show equivalence for same Bundle class" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new BundleA)
      val b = IO(new BundleA)
      assert(DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show equivalence for different Bundle classes with same fields" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new BundleA)
      val b = IO(new BundleA2)
      assert(DataMirror.checkStructuralTypeEquivalence(a, b))
      // checkTypeEquivalence requires same class, so it should fail
      assert(!DataMirror.checkTypeEquivalence(a, b))
    })
  }

  it should "show non-equivalence for different field names" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new BundleA)
      val b = IO(new BundleB)
      assert(!DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show non-equivalence for different field types" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new BundleA)
      val b = IO(new BundleC)
      assert(!DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show non-equivalence for different widths" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new BundleA)
      val b = IO(new BundleD)
      assert(!DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show equivalence for nested Records with same structure" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new NestedBundle)
      val b = IO(new NestedBundle2)
      assert(DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show equivalence for Records with Vec fields" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new VecBundle)
      val b = IO(new VecBundle2)
      assert(DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show non-equivalence for Vecs of different length" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new VecBundle)
      val b = IO(new VecBundleDiffLen)
      assert(!DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show equivalence for empty Bundles" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new EmptyBundle)
      val b = IO(new EmptyBundle2)
      assert(DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "work on unbound chisel types" in {
    val a = new BundleA
    val b = new BundleA2
    assert(DataMirror.checkStructuralTypeEquivalence(a, b))
  }

  it should "show non-equivalence between Record and non-Record types" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(new BundleA)
      val b = IO(UInt(8.W))
      assert(!DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show equivalence for Element types" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(UInt(8.W))
      val b = IO(UInt(8.W))
      assert(DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show non-equivalence for different Element types" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(UInt(8.W))
      val b = IO(SInt(8.W))
      assert(!DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show equivalence for Vec types" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Vec(4, UInt(8.W)))
      val b = IO(Vec(4, UInt(8.W)))
      assert(DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }

  it should "show non-equivalence for Vec types with different lengths" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Vec(4, UInt(8.W)))
      val b = IO(Vec(3, UInt(8.W)))
      assert(!DataMirror.checkStructuralTypeEquivalence(a, b))
    })
  }
}
