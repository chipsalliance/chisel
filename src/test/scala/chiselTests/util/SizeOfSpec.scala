// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.SizeOf
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class SizeOfBundle extends Bundle {
  val a = UInt()
  val b = SInt()
}

private class SizeOfTop extends Module {
  val io = IO(new Bundle {
    val w = Input(UInt(65.W))
    val x = Input(new SizeOfBundle)
    val outw = UInt(32.W)
    val outx = UInt(32.W)
  })
  io.outw := SizeOf(io.w)
  io.outx := SizeOf(io.x)
}

class SizeOfSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new SizeOfTop)
    println(fir)
    (fir.split('\n').map(_.trim) should contain)
      .allOf("intmodule SizeOfIntrinsic :", "input i : UInt<65>", "output size : UInt<32>", "intrinsic = circt_sizeof")
  }
}
