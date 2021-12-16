// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester

trait BundleSpecUtils {
  class BundleFooBar extends Bundle {
    val foo = UInt(16.W)
    val bar = UInt(16.W)
  }
  class BundleBarFoo extends Bundle {
    val bar = UInt(16.W)
    val foo = UInt(16.W)
  }
  class BundleFoo extends Bundle {
    val foo = UInt(16.W)
  }
  class BundleBar extends Bundle {
    val bar = UInt(16.W)
  }

  class BadSeqBundle extends Bundle {
    val bar = Seq(UInt(16.W), UInt(8.W), UInt(4.W))
  }

  class MyModule(output: Bundle, input: Bundle) extends Module {
    val io = IO(new Bundle {
      val in = Input(input)
      val out = Output(output)
    })
    io.out <> io.in
  }

  class BundleSerializationTest extends BasicTester {
    // Note that foo is higher order because its defined earlier in the Bundle
    val bundle = Wire(new BundleFooBar)
    bundle.foo := 0x1234.U
    bundle.bar := 0x5678.U
    // To UInt
    val uint = bundle.asUInt
    assert(uint.getWidth == 32) // elaboration time
    assert(uint === "h12345678".asUInt(32.W))
    // Back to Bundle
    val bundle2 = uint.asTypeOf(new BundleFooBar)
    assert(0x1234.U === bundle2.foo)
    assert(0x5678.U === bundle2.bar)
    stop()
  }
}

class BundleSpec extends ChiselFlatSpec with BundleSpecUtils with Utils {
  "Bundles with the same fields but in different orders" should "bulk connect" in {
    ChiselStage.elaborate { new MyModule(new BundleFooBar, new BundleBarFoo) }
  }

  "Bundles" should "follow UInt serialization/deserialization API" in {
    assertTesterPasses { new BundleSerializationTest }
  }

  "Bulk connect on Bundles" should "check that the fields match" in {
    (the [ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new MyModule(new BundleFooBar, new BundleFoo) }
    }).getMessage should include ("Right Record missing field")

    (the [ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new MyModule(new BundleFoo, new BundleFooBar) }
    }).getMessage should include ("Left Record missing field")
  }

  "Bundles" should "not be able to use Seq for constructing hardware" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate {
        new Module {
          val io = IO(new Bundle {
            val b = new BadSeqBundle
          })
        }
      }
    }).getMessage should include("Public Seq members cannot be used to define Bundle elements")
  }

  "Bundles" should "be allowed to have Seq if need be" in {
    assertTesterPasses {
      new BasicTester {
        val m = Module(new Module {
          val io = IO(new Bundle {
            val b = new BadSeqBundle with IgnoreSeqInBundle
          })
        })
        stop()
      }
    }
  }

  "Bundles" should "be allowed to have non-Chisel Seqs" in {
    assertTesterPasses {
      new BasicTester {
        val m = Module(new Module {
          val io = IO(new Bundle {
            val f = Output(UInt(8.W))
            val unrelated = (0 to 10).toSeq
            val unrelated2 = Seq("Hello", "World", "Chisel")
          })
          io.f := 0.U
        })
        stop()
      }
    }
  }

  "Bundles" should "with aliased fields, should show a helpful error message" in {
    class AliasedBundle extends Bundle {
      val a = UInt(8.W)
      val b = a
      val c = SInt(8.W)
      val d = c
    }

    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new Module {
        val io = IO(Output(new AliasedBundle))
        io.a := 0.U
        io.b := 1.U
      } }
    }).getMessage should include("contains aliased fields named (a,b),(c,d)")
  }

  "Bundles" should "not have bound hardware" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new Module {
        class MyBundle(val foo: UInt) extends Bundle
        val in  = IO(Input(new MyBundle(123.U))) // This should error: value passed in instead of type
        val out = IO(Output(new MyBundle(UInt(8.W))))

        out := in
      } }
    }).getMessage should include("must be a Chisel type, not hardware")
  }
  "Bundles" should "not recursively contain aggregates with bound hardware" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate { new Module {
        class MyBundle(val foo: UInt) extends Bundle
        val out = IO(Output(Vec(2, UInt(8.W))))
        val in  = IO(Input(new MyBundle(out(0)))) // This should error: Bound aggregate passed
        out := in
      } }
    }).getMessage should include("must be a Chisel type, not hardware")
  }
  "Unbound bundles sharing a field" should "not error" in {
    ChiselStage.elaborate {
      new RawModule {
        val foo = new Bundle {
          val x = UInt(8.W)
        }
        val bar = new Bundle {
          val y = foo.x
        }
      }
    }
  }

  // This tests the interaction of override def cloneType and the plugin.
  // We are commenting it for now because although this code fails to compile
  // as expected when just copied here, the test version is not seeing the failure.
  // """
  //     class BundleBaz(w: Int) extends Bundle {
  //       val baz = UInt(w.W)
  //       // This is a compiler error when using the plugin, which we test below.
  //       override def cloneType = (new BundleBaz(w)).asInstanceOf[this.type]
  //     }
  // """ shouldNot compile

}
