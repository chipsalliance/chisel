// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.NoRunFirrtlCompilerAnnotation
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
      setInline("gcd_blackbox.v", (new GCD).toVerilogString)
    }
    val bb = Module(new GCDBlackBox)
    io <> bb.io
  }

  "implicit driver" should "emit verilog without error" in {
    (new GCD).toVerilogString
  }
  "implicit driver" should "emit firrtl without error" in {
    (new GCD).toFirrtlString
  }
  "implicit driver" should "emit chirrtl without error" in {
    (new GCD).toChirrtlString
  }
  "implicit driver" should "emit system verilog without error" in {
    (new GCD).toSystemVerilogString
  }
  "implicit driver" should "compile to low firrtl" in {
    (new GCD).compile("-X", "low")
  }
  "implicit driver" should "execute with annotations" in {
    (new GCD).execute()(Seq(NoRunFirrtlCompilerAnnotation))
  }
  "implicit driver" should "work under the builder context" in {
    (new InnerModule).toVerilogString
  }
}
