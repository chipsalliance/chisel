// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.experimental.Defaulting
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals._

import scala.collection.immutable.ListMap

class DefaultingSpec extends ChiselFunSpec with Utils {
  describe("(0): Defaulting type creation") {
    it("(0.a): Creating a default type of UInt, and then accessing the value") {
      val firrtl = ChiselStage.emitFirrtl {
        new Module {
          val io = IO(Output(Defaulting(UInt(32.W), 1.U)))
          io.underlying := io.default
        }
      }
      assert(firrtl.contains("io <= UInt<1>(\"h1\")"))
    }
    it("(0.b): Creating a defaulting type of Bundle") {
      class MyBundle(width: Int) extends Bundle {
        val a = UInt(width.W)
        val b = UInt(width.W)
      }
      val firrtl = ChiselStage.emitFirrtl {
        new Module {
          val io = IO(Output(Defaulting(new MyBundle(32).Lit(_.a -> 1.U, _.b -> 2.U))))
          io.underlying := io.default
        }
      }
      assert(firrtl.contains("io.a <= UInt<1>(\"h1\")"))
      assert(firrtl.contains("io.b <= UInt<2>(\"h2\")"))
    }
    it("(0.c): Creating a defaulting type of Vec") {
      val firrtl = ChiselStage.emitFirrtl {
        new Module {
          val io = IO(Output(Defaulting(Vec(2, UInt(32.W)).Lit(0 -> 1.U, 1 -> 2.U))))
          io.underlying := io.default
        }
      }
      assert(firrtl.contains("io[0] <= UInt<32>(\"h1\")"))
      assert(firrtl.contains("io[1] <= UInt<32>(\"h2\")"))
    }
    it("(0.d): Creating a defaulting type of Record") {
      class MyRecord extends Record {
        val foo = Bool()
        val bar = Bool()
        val elements = ListMap("in" -> foo, "out" -> bar)
        def cloneType = (new MyRecord).asInstanceOf[this.type]
      }
      val firrtl = ChiselStage.emitFirrtl {
        new Module {
          val io = IO(Output(Defaulting(new MyRecord().Lit(_.elements("in") -> true.B, _.elements("out") -> false.B))))
          io.underlying := io.default
        }
      }
      assert(firrtl.contains("io.in <= UInt<1>(\"h1\")"))
      assert(firrtl.contains("io.out <= UInt<1>(\"h0\")"))
    }
  }
}
