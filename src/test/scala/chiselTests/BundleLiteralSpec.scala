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

    // Bundle literal constructor code, which will be auto-generated using macro annotations in
    // the future.
    import chisel3.core.BundleLitBinding
    import chisel3.internal.firrtl.{ULit, Width}
    // Full bundle literal constructor
    def Lit(aVal: UInt, bVal: Bool): MyBundle = {
      val clone = cloneType
      clone.selfBind(BundleLitBinding(Map(
        clone.a -> aVal.elementLitArg.get,
        clone.b -> bVal.elementLitArg.get
      )))
      clone
    }
    // Partial bundle literal constructor
    def Lit(aVal: UInt): MyBundle = {
      val clone = cloneType
      clone.selfBind(BundleLitBinding(Map(
        clone.a -> aVal.elementLitArg.get
      )))
      clone
    }
  }

  "bundle literals" should "work in RTL" in {
    val outsideBundleLit = (new MyBundle).Lit(42.U, true.B)
    assertTesterPasses{ new BasicTester{
      // TODO: add direct bundle compare operations, when that feature is added
      chisel3.assert(outsideBundleLit.a === 42.U)
      chisel3.assert(outsideBundleLit.b === true.B)

      val bundleLit = (new MyBundle).Lit(42.U, true.B)
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
      val bundleLit = (new MyBundle).Lit(42.U)
      chisel3.assert(bundleLit.a === 42.U)

      val bundleWire = Wire(new MyBundle)
      bundleWire := bundleLit

      chisel3.assert(bundleWire.a === 42.U)

      stop()
    } }
  }
}
