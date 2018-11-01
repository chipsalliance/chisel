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
    import chisel3.core.AggregateLitBinding
    import chisel3.internal.firrtl.{LitArg, ULit, Width}
    // Full bundle literal constructor
    def Lit(aVal: UInt, bVal: Bool): MyBundle = {
      val clone = cloneType
      clone.selfBind(AggregateLitBinding(clone.elements.map(_ match {
        case (n, clone.a) => (n, clone.a, litArgOfBits(aVal))
        case (n, clone.b) => (n, clone.b, litArgOfBits(bVal))
      }).toSeq))
      clone
    }
    // Partial bundle literal constructor
    def Lit(aVal: UInt): MyBundle = {
      val clone = cloneType
      clone.selfBind(AggregateLitBinding(clone.elements.flatMap(_ match {
        case (n, clone.a) => Some((n, clone.a, litArgOfBits(aVal)))
        case _ => None
      }).toSeq))
      clone
    }
  }

  class MyOuterBundle extends Bundle {
    val c = UInt(8.W)
    val d = new MyBundle

    // Bundle literal constructor code, which will be auto-generated using macro annotations in
    // the future.
    import chisel3.core.AggregateLitBinding
    import chisel3.internal.firrtl.{AggregateLit, LitArg, ULit, Width}
    // Full bundle literal constructor
    def Lit(aVal: UInt, bVal: Bool, cVal: UInt): MyOuterBundle = {
      val clone = cloneType
      AggregateLit(clone.elements.map(_ match {
        case (n, clone.c) => (n, clone.c, litArgOfBits(cVal))
        case (n, clone.d) => (n, clone.d, (new MyBundle).Lit(aVal, bVal).litArg.get)
      }).toSeq).bindLitArg(clone)
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

  "bundles inside of bundles" should "work in RTL" in {
    assertTesterPasses{ new BasicTester{
      val bundleLit = (new MyOuterBundle).Lit(42.U, true.B, 22.U)
      chisel3.assert(bundleLit.c === 22.U)
      // TODO!!!
      // chisel3.assert(bundleLit.d.a === 42.U)
      // chisel3.assert(bundleLit.d.b === true.B)

      val bundleWire = Wire(new MyOuterBundle)
      bundleWire := bundleLit

      chisel3.assert(bundleWire.c === 22.U)
      // TODO!!!
      // chisel3.assert(bundleWire.d.a === 42.U)
      // chisel3.assert(bundleWire.d.b === true.B)

      stop()
    } }
  }
}
