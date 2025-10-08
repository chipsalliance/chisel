// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.circt

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util.circt.IsX
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class IsXBundle extends Bundle {
  val a = UInt()
  val b = SInt()
}

private class IsXTop extends Module {
  val w = IO(Input(UInt(65.W)))
  val x = IO(Input(new IsXBundle))
  val y = IO(Input(UInt(65.W)))
  val outw = IO(Output(UInt(1.W)))
  val outx = IO(Output(UInt(1.W)))
  val outy = IO(Output(UInt(1.W)))
  outw := IsX(w)
  outx := IsX(x)
  outy := IsX(y)
}

class IsXSpec extends AnyFlatSpec with Matchers {
  it should "generate expected FIRRTL" in {
    val fir = ChiselStage.emitCHIRRTL(new IsXTop)
    (fir.split('\n').map(_.takeWhile(_ != '@').trim) should contain).allOf(
      "node _outw_T = intrinsic(circt_isX : UInt<1>, w)",
      "node _outx_T = intrinsic(circt_isX : UInt<1>, x)",
      "node _outy_T = intrinsic(circt_isX : UInt<1>, y)"
    )
  }
}
