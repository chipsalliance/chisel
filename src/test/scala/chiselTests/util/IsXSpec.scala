// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.testers.BasicTester
import chisel3.util.circt.IsX

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class IsXBundle extends Bundle {
  val a = UInt()
  val b = SInt()
}

private class IsXTop extends Module {
  val io = IO(new Bundle {
    val w = Input(UInt(65.W))
    val x = Input(new IsXBundle)
    val y = Input(UInt(65.W))
    val outw = UInt(1.W)
    val outx = UInt(1.W)
    val outy = UInt(1.W)
  })
  io.outw := IsX(io.w)
  io.outx := IsX(io.x)
  io.outy := IsX(io.y)
}

/** A test for intrinsics.  Since chisel is producing intrinsics as tagged
  * extmodules (for now), we explicitly test the chirrtl and annotations rather
  * than the processed firrtl or verilog.  It is worth noting that annotations
  * are implemented (for now) in a way which makes the output valid for all
  * firrtl compilers, hence we write a localized, not end-to-end test
  */
class IsXSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitChirrtl(new IsXTop)
    val a1 = """(?s)extmodule IsX_(\d+).*defname = IsX_\1.*extmodule IsX_(\d+).*defname = IsX_\2.*extmodule IsX_(\d+).*defname = IsX_\3""".r
    (fir should include).regex(a1)

    // The second elaboration uses a unique name since the Builder is reused (?)
    val c = """Intrinsic\(~IsXTop\|IsX.*,circt.isX\)"""
    ((new chisel3.stage.ChiselStage)
      .execute(
        args = Array("--no-run-firrtl"),
        annotations = Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new IsXTop))
      )
      .mkString("\n") should include).regex(c)
  }
}
