// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.circt

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.{Mux2Cell, Mux4Cell}
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class Mux2CellTop extends Module {
  val sel = IO(Input(UInt(1.W)))
  val high = IO(Input(UInt(32.W)))
  val low = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  out := Mux2Cell(sel, high, low)
}

class Mux2CellSpec extends AnyFlatSpec with Matchers {
  it should "work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new Mux2CellTop)
    (fir.split('\n').map(_.trim.takeWhile(_ != '@')) should contain)
      .allOf(
        "intmodule Mux2CellIntrinsic : ",
        "input sel : UInt<1>",
        "input high : UInt<32>",
        "input low : UInt<32>",
        "output out : UInt<32>",
        "intrinsic = circt_mux2cell"
      )
  }
}

private class Mux4CellTop extends Module {
  val sel = IO(Input(UInt(2.W)))
  val v3 = IO(Input(UInt(32.W)))
  val v2 = IO(Input(UInt(32.W)))
  val v1 = IO(Input(UInt(32.W)))
  val v0 = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  out := Mux4Cell(sel, v3, v2, v1, v0)
}

class Mux4CellSpec extends AnyFlatSpec with Matchers {
  it should "work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new Mux4CellTop)
    (fir.split('\n').map(_.trim.takeWhile(_ != '@')) should contain)
      .allOf(
        "intmodule Mux4CellIntrinsic : ",
        "input sel : UInt<2>",
        "input v3 : UInt<32>",
        "input v2 : UInt<32>",
        "input v1 : UInt<32>",
        "input v0 : UInt<32>",
        "output out : UInt<32>",
        "intrinsic = circt_mux4cell"
      )
  }
}
