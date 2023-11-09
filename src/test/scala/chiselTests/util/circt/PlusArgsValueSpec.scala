// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.circt

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.PlusArgsValue
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class PlusArgsValueTop extends Module {
  val wf = IO(Output(UInt(1.W)))
  val wv = IO(Output(UInt(32.W)))
  val xf = IO(Output(UInt(1.W)))
  val xv = IO(Output(UInt(32.W)))
  val zv = IO(Output(UInt(32.W)))
  val tmpw = PlusArgsValue(UInt(32.W), "FOO=%d")
  val tmpx = PlusArgsValue(xv, "BAR=%d")
  zv := PlusArgsValue(xv, "BAR=%d", 42.U)
  wf := tmpw.found
  wv := tmpw.result
  xf := tmpx.found
  xv := tmpx.result
}

class PlusArgsValueSpec extends AnyFlatSpec with Matchers {
  it should "work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new PlusArgsValueTop)
    (fir.split('\n').map(_.trim.takeWhile(_ != '@')) should contain).inOrder(
      "intmodule PlusArgsValueIntrinsic : ",
      "output found : UInt<1>",
      "output result : UInt<32>",
      "intrinsic = circt_plusargs_value",
      "parameter FORMAT = \"FOO=%d\"",
      "parameter FORMAT = \"BAR=%d\"",
      "node _zv_T = mux(PlusArgsValueIntrinsic_2.found, PlusArgsValueIntrinsic_2.result, UInt<6>(0h2a)) "
    )
  }
}
