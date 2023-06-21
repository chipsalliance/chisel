// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.circt

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.{MuxCell2, MuxCell4}
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class MuxCell2Top extends Module {
  val sel = IO(Input(UInt(1.W)))
  val high = IO(Input(UInt(32.W)))
  val low = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  out := MuxCell2(sel, high, low)
}

class MuxCell2Spec extends AnyFlatSpec with Matchers {
  it should "work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new MuxCell2Top)
    (fir.split('\n').map(_.trim) should contain)
      .allOf(
        "intmodule MuxCell2Intrinsic :",
        "input sel : UInt<1>",
        "input high : UInt<32>",
        "input low : UInt<32>",
        "output out : UInt<32>",
        "intrinsic = circt_muxcell2"
      )
  }
}

private class MuxCell4Top extends Module {
  val sel = IO(Input(UInt(2.W)))
  val v3 = IO(Input(UInt(32.W)))
  val v2 = IO(Input(UInt(32.W)))
  val v1 = IO(Input(UInt(32.W)))
  val v0 = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  out := MuxCell4(sel, v3, v2, v1, v0)
}

class MuxCell4Spec extends AnyFlatSpec with Matchers {
  it should "work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new MuxCell4Top)
    (fir.split('\n').map(_.trim) should contain)
      .allOf(
        "intmodule MuxCell4Intrinsic :",
        "input sel : UInt<2>",
        "input v3 : UInt<32>",
        "input v2 : UInt<32>",
        "input v1 : UInt<32>",
        "input v0 : UInt<32>",
        "output out : UInt<32>",
        "intrinsic = circt_muxcell4"
      )
  }
}
