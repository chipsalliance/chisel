// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester

trait BundleSpecUtils {
  class BundleFooBar extends Bundle {
    val foo = UInt(16.W)
    val bar = UInt(16.W)
    override def cloneType = (new BundleFooBar).asInstanceOf[this.type]
  }
  class BundleBarFoo extends Bundle {
    val bar = UInt(16.W)
    val foo = UInt(16.W)
    override def cloneType = (new BundleBarFoo).asInstanceOf[this.type]
  }
  class BundleFoo extends Bundle {
    val foo = UInt(16.W)
    override def cloneType = (new BundleFoo).asInstanceOf[this.type]
  }
  class BundleBar extends Bundle {
    val bar = UInt(16.W)
    override def cloneType = (new BundleBar).asInstanceOf[this.type]
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

class BundleSpec extends ChiselFlatSpec with BundleSpecUtils {
  "Bundles with the same fields but in different orders" should "bulk connect" in {
    elaborate { new MyModule(new BundleFooBar, new BundleBarFoo) }
  }

  "Bundles" should "follow UInt serialization/deserialization API" in {
    assertTesterPasses { new BundleSerializationTest }
  }

  "Bulk connect on Bundles" should "check that the fields match" in {
    (the [ChiselException] thrownBy {
      elaborate { new MyModule(new BundleFooBar, new BundleFoo) }
    }).getMessage should include ("Right Record missing field")

    (the [ChiselException] thrownBy {
      elaborate { new MyModule(new BundleFoo, new BundleFooBar) }
    }).getMessage should include ("Left Record missing field")
  }
}
