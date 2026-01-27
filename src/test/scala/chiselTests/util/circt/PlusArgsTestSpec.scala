// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.circt

import chisel3._
import chisel3.util.circt.PlusArgsTest
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.io.Source

private class PlusArgsTestTop extends Module {
  val w = IO(Output(UInt(1.W)))
  val x = IO(Output(UInt(1.W)))
  w := PlusArgsTest("FOO")
  x := PlusArgsTest("BAR")
}

class PlusArgsTestSpec extends AnyFlatSpec with Matchers {
  it should "generate expected FIRRTL" in {
    val fir = ChiselStage.emitCHIRRTL(new PlusArgsTestTop)
    (fir.split('\n').map(_.takeWhile(_ != '@').trim) should contain).allOf(
      """node _w_T = intrinsic(circt_plusargs_test<FORMAT = "FOO"> : UInt<1>)""",
      """node _x_T = intrinsic(circt_plusargs_test<FORMAT = "BAR"> : UInt<1>)"""
    )
  }
  it should "compile to SV" in {
    ChiselStage.emitSystemVerilog(new PlusArgsTestTop)
  }
}
