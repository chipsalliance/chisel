// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.Cat
import circt.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals.AddVecLiteralConstructor
import chisel3.experimental.BundleLiteralException

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

  class LongBundle extends Bundle {
    val a = UInt(48.W)
    val b = SInt(32.W)
    val c = UInt(16.W)
  }

  class ChildBundle extends Bundle {
    val foo = UInt(4.W)
  }
  class ComplexBundle(w: Int) extends Bundle {
    val a = Vec(2, new ChildBundle)
    val b = UInt(w.W)
    val c = UInt(4.W)
  }

  "bundle literals" should "pack" in {
    assertTesterPasses {
      new BasicTester {
        val bundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sB)
        bundleLit.litOption should equal(Some(169)) // packed as 42 (8-bit), false=0 (1-bit), sB=1 (1-bit)
        chisel3.assert(bundleLit.asUInt === bundleLit.litOption.get.U) // sanity-check consistency with runtime

        val longBundleLit =
          (new LongBundle).Lit(_.a -> 0xdeaddeadbeefL.U, _.b -> (-0x0beef00dL).S(32.W), _.c -> 0xfafa.U)
        longBundleLit.litOption should equal(
          Some(
            (BigInt(0xdeaddeadbeefL) << 48)
              + (BigInt(0xffffffffL - 0xbeef00dL + 1) << 16)
              + BigInt(0xfafa)
          )
        )
        chisel3.assert(longBundleLit.asUInt === longBundleLit.litOption.get.U)

        stop()
      }
    }
  }

  "bundle literals" should "work in RTL" in {
    val outsideBundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
    assertTesterPasses {
      new BasicTester {
        // TODO: add direct bundle compare operations, when that feature is added
        chisel3.assert(outsideBundleLit.a === 42.U)
        chisel3.assert(outsideBundleLit.b === true.B)
        chisel3.assert(outsideBundleLit.c === MyEnum.sB)
        chisel3.assert(outsideBundleLit.isLit)
        chisel3.assert(outsideBundleLit.litValue.U === outsideBundleLit.asUInt)
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
      }
    }
  }

  "bundle literals of vec literals" should "work" in {
    assertTesterPasses(new BasicTester {
      val bundleWithVecs = new Bundle {
        val a = Vec(2, UInt(4.W))
        val b = Vec(2, SInt(4.W))
      }.Lit(
        _.a -> Vec(2, UInt(4.W)).Lit(0 -> 0xa.U, 1 -> 0xb.U),
        _.b -> Vec(2, SInt(4.W)).Lit(0 -> 1.S, 1 -> (-1).S)
      )
      chisel3.assert(bundleWithVecs.a(0) === 0xa.U)
      chisel3.assert(bundleWithVecs.a(1) === 0xb.U)
      chisel3.assert(bundleWithVecs.b(0) === 1.S)
      chisel3.assert(bundleWithVecs.b(1) === (-1).S)
      stop()
    })
  }

  "partial bundle literals" should "work in RTL" in {
    assertTesterPasses {
      new BasicTester {
        val bundleLit = (new MyBundle).Lit(_.a -> 42.U)
        chisel3.assert(bundleLit.a === 42.U)

        val bundleWire = Wire(new MyBundle)
        bundleWire := bundleLit

        chisel3.assert(bundleWire.a === 42.U)

        stop()
      }
    }
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
    assertTesterPasses {
      new BasicTester {
        // Specify the inner Bundle value as a Bundle literal
        val explicitBundleLit = (new MyOuterBundle).Lit(
          _.a -> (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
        )
        chisel3.assert(explicitBundleLit.a.a === 42.U)
        chisel3.assert(explicitBundleLit.a.b === true.B)
        chisel3.assert(explicitBundleLit.a.c === MyEnum.sB)
        chisel3.assert(explicitBundleLit.a.isLit)
        chisel3.assert(explicitBundleLit.a.litValue.U === explicitBundleLit.a.asUInt)

        // Specify the inner Bundle fields directly
        val expandedBundleLit = (new MyOuterBundle).Lit(
          _.a.a -> 42.U,
          _.a.b -> true.B,
          _.b.c -> false.B,
          _.b.d -> 255.U,
          _.b.e -> MyEnum.sB,
          _.f -> MyEnum.sB
        )
        chisel3.assert(expandedBundleLit.a.a === 42.U)
        chisel3.assert(expandedBundleLit.a.b === true.B)
        chisel3.assert(expandedBundleLit.f === MyEnum.sB)
        chisel3.assert(expandedBundleLit.b.c === false.B)
        chisel3.assert(expandedBundleLit.b.d === 255.U)
        chisel3.assert(expandedBundleLit.b.e === MyEnum.sB)
        chisel3.assert(!expandedBundleLit.a.isLit) // element e is missing
        chisel3.assert(expandedBundleLit.b.isLit)
        chisel3.assert(!expandedBundleLit.isLit) // element a.e is missing
        chisel3.assert(expandedBundleLit.b.litValue.U === expandedBundleLit.b.asUInt)

        // Anonymously contruct the inner Bundle literal
        // A bit of weird syntax that depends on implementation details of the Bundle literal constructor
        val childBundleLit =
          (new MyOuterBundle).Lit(b => b.b -> b.b.Lit(_.c -> false.B, _.d -> 255.U, _.e -> MyEnum.sB))
        chisel3.assert(childBundleLit.b.c === false.B)
        chisel3.assert(childBundleLit.b.d === 255.U)
        chisel3.assert(childBundleLit.b.e === MyEnum.sB)
        chisel3.assert(childBundleLit.b.isLit)
        chisel3.assert(!childBundleLit.isLit) // elements a and f are missing
        chisel3.assert(childBundleLit.b.litValue.U === childBundleLit.b.asUInt)

        stop()
      }
    }
  }

  "Bundle literals" should "assign" in {
    assertTesterPasses {
      new BasicTester {
        val bundleWire = Wire(Output(new MyBundle))
        val bundleLit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
        bundleWire := bundleLit

        chisel3.assert(bundleWire.a === 42.U)
        chisel3.assert(bundleWire.b === true.B)
        chisel3.assert(bundleWire.c === MyEnum.sB)
        stop()
      }
    }
  }

  "partially initialized Bundle literals" should "assign" in {
    assertTesterPasses {
      new BasicTester {
        val bundleWire = Wire(Output(new MyBundle))
        val bundleLit = (new MyBundle).Lit(_.a -> 42.U)
        bundleWire := bundleLit

        chisel3.assert(bundleWire.a === 42.U)
        stop()
      }
    }
  }

  "Bundle literals" should "work as register reset values" in {
    assertTesterPasses {
      new BasicTester {
        val r = RegInit((new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB))
        r := (r.asUInt + 1.U).asTypeOf(new MyBundle) // prevent constprop

        // check reset values on first cycle out of reset
        chisel3.assert(r.a === 42.U)
        chisel3.assert(r.b === true.B)
        chisel3.assert(r.c === MyEnum.sB)
        stop()
      }
    }
  }

  "partially initialized Bundle literals" should "work as register reset values" in {
    assertTesterPasses {
      new BasicTester {
        val r = RegInit((new MyBundle).Lit(_.a -> 42.U))
        r.a := r.a + 1.U // prevent const prop
        chisel3.assert(r.a === 42.U) // coming out of reset
        stop()
      }
    }
  }

  "Fields extracted from BundleLiterals" should "work as register reset values" in {
    assertTesterPasses {
      new BasicTester {
        val r = RegInit((new MyBundle).Lit(_.a -> 42.U).a)
        r := r + 1.U // prevent const prop
        chisel3.assert(r === 42.U) // coming out of reset
        stop()
      }
    }
  }

  "DontCare fields extracted from BundleLiterals" should "work as register reset values" in {
    assertTesterPasses {
      new BasicTester {
        val r = RegInit((new MyBundle).Lit(_.a -> 42.U).b)
        r := reset.asBool
        printf(p"r = $r\n") // Can't assert because reset value is DontCare
        stop()
      }
    }
  }

  "DontCare fields extracted from BundleLiterals" should "work in other Expressions" in {
    assertTesterPasses {
      new BasicTester {
        val x = (new MyBundle).Lit(_.a -> 42.U).b || true.B
        chisel3.assert(x === true.B)
        stop()
      }
    }
  }

  "bundle literals with bad field specifiers" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.emitCHIRRTL {
          new RawModule {
            val bundle = new MyBundle
            bundle.Lit(x => bundle.a -> 0.U) // DONT DO THIS, this gets past a syntax error to exercise the failure
          }
        }
      }
    }
    exc.getMessage should include("not a field")
  }

  "bundle literals with duplicate fields" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.emitCHIRRTL {
          new RawModule {
            (new MyBundle).Lit(_.a -> 0.U, _.a -> 0.U)
          }
        }
      }
    }
    exc.getMessage should include("duplicate")
    exc.getMessage should include(".a")
  }

  "bundle literals with non-literal values" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.emitCHIRRTL {
          new RawModule {
            (new MyBundle).Lit(_.a -> UInt())
          }
        }
      }
    }
    exc.getMessage should include("non-literal value")
    exc.getMessage should include(".a")
  }

  "bundle literals with non-type-equivalent element fields" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.emitCHIRRTL {
          new RawModule {
            (new MyBundle).Lit(_.a -> true.B)
          }
        }
      }
    }
    exc.getMessage should include("non-type-equivalent value")
    exc.getMessage should include(".a")
  }

  "bundle literals with non-type-equivalent sub-bundles" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.emitCHIRRTL {
          new RawModule {
            (new MyOuterBundle).Lit(_.b -> (new MyBundle).Lit(_.a -> 0.U))
          }
        }
      }
    }
    exc.getMessage should include("non-type-equivalent value")
    exc.getMessage should include(".b")
  }

  "bundle literals with non-type-equivalent enum element fields" should "fail" in {
    val exc = intercept[BundleLiteralException] {
      extractCause[BundleLiteralException] {
        ChiselStage.emitCHIRRTL {
          new RawModule {
            (new MyBundle).Lit(_.c -> MyEnumB.sB)
          }
        }
      }
    }
    exc.getMessage should include("non-type-equivalent enum value")
    exc.getMessage should include(".c")
  }

  "bundle literals with too-wide of literal values" should "warn and truncate" in {
    class SimpleBundle extends Bundle {
      val a = UInt(4.W)
      val b = UInt(4.W)
    }
    val (stdout, _, chirrtl) = grabStdOutErr(ChiselStage.emitCHIRRTL(new RawModule {
      val lit = (new SimpleBundle).Lit(_.a -> 0xde.U, _.b -> 0xad.U)
      val x = Cat(lit.a, lit.b)
    }))
    stdout should include("[W007] Literal value ULit(222,) is too wide for field _.a with width 4")
    stdout should include("[W007] Literal value ULit(173,) is too wide for field _.b with width 4")
    chirrtl should include("node x = cat(UInt<4>(0he), UInt<4>(0hd))")
  }

  "bundle literals with zero-width fields" should "not warn for 0.U" in {
    class SimpleBundle extends Bundle {
      val a = UInt(4.W)
      val b = UInt(0.W)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(
      new RawModule {
        val lit = (new SimpleBundle).Lit(_.a -> 5.U, _.b -> 0.U)
        val x = Cat(lit.a, lit.b)
      },
      args = Array("--warnings-as-errors")
    )
    chirrtl should include("node x = cat(UInt<4>(0h5), UInt<0>(0h0))")
  }

  "partial bundle literals" should "fail to pack" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val bundleLit = (new MyBundle).Lit(_.a -> 42.U)
        bundleLit.litOption should equal(None)
      }
    }
  }

  "bundle literals" should "materialize const wires" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val r = RegInit((new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB))
    })
    val wire = """wire.*: const \{ a : UInt<8>, b : UInt<1>, c : UInt<1>\}""".r
    (chirrtl should include).regex(wire)
  }

  "Empty bundle literals" should "be supported" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val lit = (new Bundle {}).Lit()
      lit.litOption should equal(Some(0))
    })
  }

  "bundle literals" should "use the widths of the Bundle fields rather than the widths of the literals" in {
    class SimpleBundle extends Bundle {
      val a = UInt(4.W)
      val b = UInt(4.W)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      // Whether the user specifies a width or not.
      val lit = (new SimpleBundle).Lit(_.a -> 0x3.U, _.b -> 0x3.U(3.W))
      lit.a.getWidth should be(4)
      lit.b.getWidth should be(4)
      val cat = Cat(lit.a, lit.b)
    })
    chirrtl should include("node cat = cat(UInt<4>(0h3), UInt<4>(0h3))")
  }

  "Calling .asUInt on a Bundle literal" should "return a UInt literal and work outside of elaboration" in {
    val blit = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B, _.c -> MyEnum.sB)
    val ulit = blit.asUInt
    ulit.litOption should be(Some(171))

    assertTesterPasses {
      new BasicTester {
        // Check that it gives the same value as the generated hardware
        val wire = WireInit(blit).asUInt
        chisel3.assert(ulit.litValue.U === wire)
        stop()
      }
    }
  }

  "Calling .asUInt on a Bundle literal with DontCare fields" should "NOT return a UInt literal" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val blit = (new MyBundle).Lit(_.a -> 42.U, _.c -> MyEnum.sB)
      val ulit = blit.asUInt
      ulit.litOption should be(None)
    })
  }

  "Calling .asUInt on a Bundle literal with zero-width fields" should "return a UInt literal and work outside of elaboration" in {
    import chisel3.experimental.VecLiterals._

    val vlit = Vec.Lit((new ChildBundle).Lit(_.foo -> 0xa.U), (new ChildBundle).Lit(_.foo -> 0xb.U))
    val blit = (new ComplexBundle(0)).Lit(_.a -> vlit, _.b -> 0.U(0.W), _.c -> 0xc.U)
    val ulit = blit.asUInt
    ulit.litOption should be(Some(0xbac))

    assertTesterPasses {
      new BasicTester {
        // Check that it gives the same value as the generated hardware
        val wire = WireInit(blit).asUInt
        chisel3.assert(ulit.litValue.U === wire)
        stop()
      }
    }
  }

}
