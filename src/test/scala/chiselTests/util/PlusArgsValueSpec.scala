// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.PlusArgsValue
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class PlusArgsValueTop extends Module {
  val io = IO(new Bundle {
    val wf = Output(UInt(1.W))
    val wv = Output(UInt(32.W))
    val xf = Output(UInt(1.W))
    val xv = Output(UInt(32.W))
  })
  val tmpw = PlusArgsValue(UInt(32.W), "FOO=%d")
  val tmpx = PlusArgsValue(io.xv, "BAR=%d")
  io.xf := tmpx.found
  io.xv := tmpx.result
}

/** A test for intrinsics.  Since chisel is producing intrinsics as tagged
  * extmodules (for now), we explicitly test the chirrtl and annotations rather
  * than the processed firrtl or verilog.  It is worth noting that annotations
  * are implemented (for now) in a way which makes the output valid for all
  * firrtl compilers, hence we write a localized, not end-to-end test
  */
class PlusArgsValueSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new PlusArgsValueTop)
    println(fir)
    (fir.split('\n').map(x => x.trim) should contain).inOrder(
      "intmodule PlusArgsValueIntrinsic :",
      "output found : UInt<1>",
      "output result : UInt<32>",
      "intrinsic = circt.plusargs.value",
      "parameter FORMAT = \"FOO=%d\"",
      "parameter FORMAT = \"BAR=%d\""
    )
  }
}
