// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.BundleLiteralException
import chisel3.experimental.ChiselEnum

class BundleLiteralSpec extends ChiselFlatSpec with Utils {
  object MyEnum extends ChiselEnum {
    val sA, sB = Value
  }
  object MyEnumB extends ChiselEnum {
    val sA, sB = Value
  }
  class MyBundle extends Bundle {
    val a = UInt(8.W)
    val b = Bool()
    val c = MyEnum()
  }

  "bundle literals" should "work in RTL" in {
    val outsideBundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
    assertTesterPasses{ new BasicTester{
      // TODO: add direct bundle compare operations, when that feature is added
      chisel3.assert(outsideBundleLit.a === 42.U)
      chisel3.assert(outsideBundleLit.b === true.B)
      chisel3.assert(outsideBundleLit.c === MyEnum.sB)

      val bundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
      chisel3.assert(bundleLit.a === 42.U)
      chisel3.assert(bundleLit.b === true.B)
      chisel3.assert(bundleLit.c === MyEnum.sB)

      chisel3.assert(bundleLit.a === outsideBundleLit.a)
      chisel3.assert(bundleLit.b === outsideBundleLit.b)
      chisel3.assert(bundleLit.c === outsideBundleLit.c)

      val bundleWire = Wire(new MyBundle)
      bundleWire := outsideBundleLit

      chisel3.assert(bundleWire.a === 42.U)
      chisel3.assert(bundleWire.b === true.B)
      chisel3.assert(bundleWire.c === MyEnum.sB)

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
      val e = MyEnum()
    }
    val f = MyEnum()
  }

  "contained bundles" should "work" in {
    assertTesterPasses{ new BasicTester{
      // Specify the inner Bundle value as a Bundle literal
      val explicitBundleLit = (new MyOuterBundle).Lit(
        _.a -> (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
      )
      chisel3.assert(explicitBundleLit.a.a === 42.U)
      chisel3.assert(explicitBundleLit.a.b === true.B)
      chisel3.assert(explicitBundleLit.a.c === MyEnum.sB)

      // Specify the inner Bundle fields directly
      val expandedBundleLit = (new MyOuterBundle).Lit(
        _.a.a -> 42.U, _.a.b -> true.B,
        _.b.c -> false.B, _.b.d -> 255.U, _.b.e -> MyEnum.sB,
        _.f -> MyEnum.sB
      )
      chisel3.assert(expandedBundleLit.a.a === 42.U)
      chisel3.assert(expandedBundleLit.a.b === true.B)
      chisel3.assert(expandedBundleLit.f === MyEnum.sB)
      chisel3.assert(expandedBundleLit.b.c === false.B)
      chisel3.assert(expandedBundleLit.b.d === 255.U)
      chisel3.assert(expandedBundleLit.b.e === MyEnum.sB)

      // Anonymously contruct the inner Bundle literal
      // A bit of weird syntax that depends on implementation details of the Bundle literal constructor
      val childBundleLit = (new MyOuterBundle).Lit(
        b => b.b -> b.b.Lit(_.c -> false.B, _.d -> 255.U, _.e -> MyEnum.sB)
      )
      chisel3.assert(childBundleLit.b.c === false.B)
      chisel3.assert(childBundleLit.b.d === 255.U)
      chisel3.assert(childBundleLit.b.e === MyEnum.sB)

      stop()
    } }
  }

  "Bundle literals" should "assign" in {
    assertTesterPasses{ new BasicTester{
      val bundleWire = Wire(Output(new MyBundle))
      val bundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
      bundleWire := bundleLit

      chisel3.assert(bundleWire.a === 42.U)
      chisel3.assert(bundleWire.b === true.B)
      chisel3.assert(bundleWire.c === MyEnum.sB)
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
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.elaborate {
          new RawModule {
            val bundle = new MyBundle
            bundle.Lit(x => bundle.a -> 0.U)  // DONT DO THIS, this gets past a syntax error to exercise the failure
          }
        }
      }
    }
    exc.getMessage should include ("not a field")
  }

  "bundle literals with duplicate fields" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.elaborate {
          new RawModule {
            (new MyBundle).Lit(_.a -> 0.U, _.a -> 0.U)
          }
        }
      }
    }
    exc.getMessage should include ("duplicate")
    exc.getMessage should include (".a")
  }

  "bundle literals with non-literal values" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.elaborate { new RawModule {
                     (new MyBundle).Lit(_.a -> UInt())
                   }
        }
      }
    }
    exc.getMessage should include ("non-literal value")
    exc.getMessage should include (".a")
  }

  "bundle literals with non-type-equivalent element fields" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.elaborate {
          new RawModule {
            (new MyBundle).Lit(_.a -> true.B)
          }
        }
      }
    }
    exc.getMessage should include ("non-type-equivalent value")
    exc.getMessage should include (".a")
  }

  "bundle literals with non-type-equivalent sub-bundles" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.elaborate {
          new RawModule {
            (new MyOuterBundle).Lit(_.b -> (new MyBundle).Lit(_.a -> 0.U))
          }
        }
      }
    }
    exc.getMessage should include ("non-type-equivalent value")
    exc.getMessage should include (".b")
  }

  "bundle literals with non-type-equivalent enum element fields" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.elaborate {
          new RawModule {
            (new MyBundle).Lit(_.c -> MyEnumB.sB)
          }
        }
      }
    }
    exc.getMessage should include ("non-type-equivalent enum value")
    exc.getMessage should include (".c")
  }

}
