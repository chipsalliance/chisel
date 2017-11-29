// See LICENSE for license details.

package chiselTests
import Chisel.ChiselException
import org.scalatest._
import chisel3._

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

  behavior of "anonymous doubly-nested inner bundle fails with clear error"
  ( the[ChiselException] thrownBy {
    elaborate {
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
    }
  }).getMessage should include("Unable to determine instance")

}
