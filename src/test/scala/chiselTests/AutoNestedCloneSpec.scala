// SPDX-License-Identifier: Apache-2.0

package chiselTests
import chisel3._
import chisel3.testers.TestUtils
import org.scalatest.matchers.should.Matchers

class BundleWithAnonymousInner(val w: Int) extends Bundle {
  val inner = new Bundle {
    val foo = Input(UInt(w.W))
  }
}

// TODO all `.suggestNames` are due to https://github.com/chipsalliance/chisel3/issues/1802
class AutoNestedCloneSpec extends ChiselFlatSpec with Matchers with Utils {
  val usingPlugin: Boolean = TestUtils.usingPlugin
  val elaborate = TestUtils.elaborateNoReflectiveAutoCloneType _

  behavior of "autoCloneType of inner Bundle in Chisel3"

  it should "clone a doubly-nested inner bundle successfully" in {
    elaborate {
      class Outer(val w: Int) extends Module {
        class Middle(val w: Int) {
          class InnerIOType extends Bundle {
            val in = Input(UInt(w.W))
          }
          def getIO: InnerIOType = new InnerIOType
        }
        val io = IO(new Bundle {}).suggestName("io")
        val myWire = Wire((new Middle(w)).getIO)
      }
      new Outer(2)
    }
  }

  it should "clone an anonymous inner bundle successfully" in {
    elaborate {
      class TestTop(val w: Int) extends Module {
        val io = IO(new Bundle {}).suggestName("io")
        val myWire = Wire(new Bundle{ val a = UInt(w.W) })
      }
      new TestTop(2)
    }
  }

  it should "pick the correct $outer instance for an anonymous inner bundle" in {
    elaborate {
      class Inner(val w: Int) extends Module {
        val io = IO(new Bundle{
          val in = Input(UInt(w.W))
          val out = Output(UInt(w.W))
        }).suggestName("io")
      }
      class Outer(val w: Int) extends Module {
        val io = IO(new Bundle{
          val in = Input(UInt(w.W))
          val out = Output(UInt(w.W))
        }).suggestName("io")
        val i = Module(new Inner(w))
        val iw = Wire(chiselTypeOf(i.io))
        iw <> io
        i.io <> iw
      }
      new Outer(2)
    }
  }

  it should "clone an anonymous, bound, inner bundle of another bundle successfully" in {
    elaborate {
      class TestModule(w: Int) extends Module {
        val io = IO(new BundleWithAnonymousInner(w) ).suggestName("io")
        val w0 = WireDefault(io)
        val w1 = WireDefault(io.inner)
      }
      new TestModule(8)
    }
  }

  it should "clone an anonymous, inner bundle of a Module, bound to another bundle successfully" in {
    elaborate {
      class TestModule(w: Int) extends Module {
        val bun = new Bundle {
          val foo = UInt(w.W)
        }
        val io = IO(new Bundle {
          val inner = Input(bun)
        }).suggestName("io")
        val w0 = WireDefault(io)
        val w1 = WireDefault(io.inner)
      }
      new TestModule(8)
    }
  }

  it should "clone a double-nested anonymous Bundle" in {
    elaborate {
      class TestModule() extends Module {
        val io = IO(new Bundle {
          val inner = Input(new Bundle {
            val x = UInt(8.W)
          })
        }).suggestName("io")
      }
      new TestModule()
    }
  }

  if (usingPlugin) {
    // This works with the plugin, but is a null pointer exception when using reflective autoclonetype
    it should "support an anonymous doubly-nested inner bundle" in {
      elaborate {
        class Outer(val w: Int) extends Module {
          class Middle(val w: Int) {
            def getIO: Bundle = new Bundle {
              val in = Input(UInt(w.W))
            }
          }
          val io = IO(new Bundle {}).suggestName("io")
          val myWire = Wire((new Middle(w)).getIO)
        }
        new Outer(2)
      }
    }

    it should "support anonymous Inner bundles that capture type parameters from outer Bundles" in {
      elaborate(new MultiIOModule {
        class MyBundle[T <: Data](n: Int, gen: T) extends Bundle {
          val foo = new Bundle {
            val x = Input(Vec(n, gen))
          }
          val bar = Output(Option(new { def mkBundle = new Bundle { val x = Vec(n, gen) }}).get.mkBundle)
        }
        val io = IO(new MyBundle(4, UInt(8.W)))
        val myWire = WireInit(io.foo)
        val myWire2 = WireInit(io.bar)
        io.bar.x := io.foo.x
      })
    }
  }
}
