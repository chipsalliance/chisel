// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.experimental.RawModule
import chisel3.experimental.BundleLiterals._
import chisel3.core.BundleLiteralException

class BundleLiteralSpec extends ChiselFlatSpec {
  class MyBundle extends Bundle {
    val a = UInt(8.W)
    val b = Bool()
  }

  "bundle literals" should "work in RTL" in {
    val outsideBundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B)
    assertTesterPasses{ new BasicTester{
      // TODO: add direct bundle compare operations, when that feature is added
      chisel3.assert(outsideBundleLit.a === 42.U)
      chisel3.assert(outsideBundleLit.b === true.B)

      val bundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B)
      chisel3.assert(bundleLit.a === 42.U)
      chisel3.assert(bundleLit.b === true.B)

      chisel3.assert(bundleLit.a === outsideBundleLit.a)
      chisel3.assert(bundleLit.b === outsideBundleLit.b)

      val bundleWire = Wire(new MyBundle)
      bundleWire := outsideBundleLit

      chisel3.assert(bundleWire.a === 42.U)
      chisel3.assert(bundleWire.b === true.B)

      stop()
    } }
  }

  "partial bundle literals" should "work in RTL" in {
    assertTesterPasses{ new BasicTester{
      val bundleLit = (new MyBundle).Lit(_.a -> 42.U)
      chisel3.assert(bundleLit.a === 42.U)

      val bundleWire = Wire(new MyBundle)
      bundleWire := bundleLit

      chisel3.assert(bundleWire.a === 42.U)

      stop()
    } }
  }

  class MyOuterBundle extends Bundle {
    val a = new MyBundle
    val b = new Bundle {
      val c = Bool()
      val d = UInt(8.W)
    }
  }

  "contained bundles" should "work" in {
    assertTesterPasses{ new BasicTester{
      // Specify the inner Bundle value as a Bundle literal
      val explicitBundleLit = (new MyOuterBundle).Lit(
        _.a -> (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B)
      )
      chisel3.assert(explicitBundleLit.a.a === 42.U)
      chisel3.assert(explicitBundleLit.a.b === true.B)

      // Specify the inner Bundle fields directly
      val expandedBundleLit = (new MyOuterBundle).Lit(
        _.a.a -> 42.U, _.a.b -> true.B,
        _.b.c -> false.B, _.b.d -> 255.U
      )
      chisel3.assert(expandedBundleLit.a.a === 42.U)
      chisel3.assert(expandedBundleLit.a.b === true.B)
      chisel3.assert(expandedBundleLit.b.c === false.B)
      chisel3.assert(expandedBundleLit.b.d === 255.U)

      // Anonymously contruct the inner Bundle literal
      // A bit of weird syntax that depends on implementation details of the Bundle literal constructor
      val childBundleLit = (new MyOuterBundle).Lit(
        b => b.b -> b.b.Lit(_.c -> false.B, _.d -> 255.U)
      )
      chisel3.assert(childBundleLit.b.c === false.B)
      chisel3.assert(childBundleLit.b.d === 255.U)

      stop()
    } }
  }

  "Bundle literals" should "assign" in {
    assertTesterPasses{ new BasicTester{
      val bundleWire = Wire(Output(new MyBundle))
      val bundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B)
      bundleWire := bundleLit

      chisel3.assert(bundleWire.a === 42.U)
      chisel3.assert(bundleWire.b === true.B)
      stop()
    } }
  }

  "partially initialized Bundle literals" should "assign" in {
    assertTesterPasses{ new BasicTester{
      val bundleWire = Wire(Output(new MyBundle))
      val bundleLit = (new MyBundle).Lit(_.a -> 42.U)
      bundleWire := bundleLit

      chisel3.assert(bundleWire.a === 42.U)
      stop()
    } }
  }

  "bundle literals with bad field specifiers" should "fail" in {
    val exc = intercept[BundleLiteralException] { elaborate { new RawModule {
      val bundle = new MyBundle
      bundle.Lit(x => bundle.a -> 0.U)  // DONT DO THIS, this gets past a syntax error to exercise the failure
    }}}
    exc.getMessage should include ("not a field")
  }

  "bundle literals with duplicate fields" should "fail" in {
    val exc = intercept[BundleLiteralException] { elaborate { new RawModule {
      (new MyBundle).Lit(_.a -> 0.U, _.a -> 0.U)
    }}}
    exc.getMessage should include ("duplicate")
    exc.getMessage should include (".a")
  }

  "bundle literals with non-literal values" should "fail" in {
    val exc = intercept[BundleLiteralException] { elaborate { new RawModule {
      (new MyBundle).Lit(_.a -> UInt())
    }}}
    exc.getMessage should include ("non-literal value")
    exc.getMessage should include (".a")
  }

  "bundle literals with non-type-equivalent element fields" should "fail" in {
    val exc = intercept[BundleLiteralException] { elaborate { new RawModule {
      (new MyBundle).Lit(_.a -> true.B)
    }}}
    exc.getMessage should include ("non-type-equivalent value")
    exc.getMessage should include (".a")
  }

  "bundle literals with non-type-equivalent sub-bundles" should "fail" in {
    val exc = intercept[BundleLiteralException] { elaborate { new RawModule {
      (new MyOuterBundle).Lit(_.b -> (new MyBundle).Lit(_.a -> 0.U))
    }}}
    exc.getMessage should include ("non-type-equivalent value")
    exc.getMessage should include (".b")
  }
}
