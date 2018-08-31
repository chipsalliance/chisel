// See LICENSE for license details.

package chiselTests
import Chisel.ChiselException
import org.scalatest._
import chisel3._

class BundleWithAnonymousInner(val w: Int) extends Bundle {
  val inner = new Bundle {
    val foo = Input(UInt(w.W))
  }
}

class AutoNestedCloneSpec extends ChiselFlatSpec with Matchers {
  behavior of "autoCloneType of inner Bundle in Chisel3"

  it should "clone a doubly-nested inner bundle successfully" in {
    elaborate {
      class Outer(val w: Int) extends Module {
        class Middle(val w: Int) {
          class InnerIOType extends Bundle {
            val in = Input(UInt(w.W))
          }
          def getIO = new InnerIOType
        }
        val io = IO(new Bundle {})
        val myWire = Wire((new Middle(w)).getIO)
      }
      new Outer(2)
    }
  }

  it should "clone an anonymous inner bundle successfully" in {
    elaborate {
      class TestTop(val w: Int) extends Module {
        val io = IO(new Bundle {})
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
        })
      }
      class Outer(val w: Int) extends Module {
        val io = IO(new Bundle{
          val in = Input(UInt(w.W))
          val out = Output(UInt(w.W))
        })
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
        val io = IO(new BundleWithAnonymousInner(w) )
        val w0 = WireInit(io)
        val w1 = WireInit(io.inner)
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
        })
        val w0 = WireInit(io)
        val w1 = WireInit(io.inner)
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
        })
      }
      new TestModule()
    }
  }

  // Test ignored because the compatibility null-inserting autoclonetype doesn't trip this
  ignore should "fail on anonymous doubly-nested inner bundle with clear error" in {
    intercept[ChiselException] { elaborate {
      class Outer(val w: Int) extends Module {
        class Middle(val w: Int) {
          def getIO = new Bundle {
            val in = Input(UInt(w.W))
          }
        }
        val io = IO(new Bundle {})
        val myWire = Wire((new Middle(w)).getIO)
      }
      new Outer(2)
    }}.getMessage should include("Unable to determine instance")
  }
}
