// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.FixedPoint
import chisel3.experimental.RawModule
import chisel3.testers.BasicTester
import org.scalatest._

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
//      val explicitBundleLit = (new MyOuterBundle).Lit(_.a -> (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B))
//      chisel3.assert(explicitBundleLit.a.a === 42.U)
//      chisel3.assert(explicitBundleLit.a.b === true.B)

      val expandedBundleLit = (new MyOuterBundle).Lit(
        _.a.a -> 42.U, _.a.b -> true.B,
        _.b.c -> false.B, _.b.d -> 244.U
      )
      chisel3.assert(expandedBundleLit.a.a === 42.U)
      chisel3.assert(expandedBundleLit.a.b === true.B)
      chisel3.assert(expandedBundleLit.b.c === false.B)
      chisel3.assert(expandedBundleLit.b.d === 255.U)

      stop()
    } }
  }
}
