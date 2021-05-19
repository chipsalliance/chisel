// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.HasBlackBoxInline
import chisel3.util.experimental.ImplicitDriver
import org.scalatest.flatspec.AnyFlatSpec

class ImplicitDriverSpec extends AnyFlatSpec {
  class InnerModule extends Module {
    class GCDIO extends Bundle {
      val a = Input(UInt(32.W))
      val b = Input(UInt(32.W))
      val e = Input(Bool())
      val z = Output(UInt(32.W))
      val v = Output(Bool())
    }
    val io = IO(new GCDIO)
    class GCDBlackBox extends BlackBox with HasBlackBoxInline {
      val io = IO(new GCDIO)
      setInline("gcd_blackbox.v", (new GCD).emitVerilog)
    }
    val bb = Module(new GCDBlackBox)
    io <> bb.io
  }

  "implicit driver" should "emit verilog without error" in {
    (new GCD).emitVerilog
  }
  "implicit driver" should "emit firrtl without error" in {
    (new GCD).emitFirrtl
  }
  "implicit driver" should "emit chirrtl without error" in {
    (new GCD).emitChirrtl
  }
  "implicit driver" should "emit system verilog without error" in {
    (new GCD).emitSystemVerilog
  }
  "implicit driver" should "work under the builder context" in {
    (new InnerModule).emitVerilog
  }
}
