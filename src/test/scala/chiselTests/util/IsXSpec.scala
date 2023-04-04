// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.testers.BasicTester
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

class IsXSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new IsXTop)
    println(fir)
    (
      (fir.split('\n').map(x => x.trim) should contain).allOf(
        "intmodule IsXIntrinsic :",
        "input i : UInt<65>",
        "output found : UInt<1>",
        "intrinsic = circt.isX",
        "input i : { a : UInt, b : SInt}",
      )
    )
  }
}
