// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.PlusArgsTest
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class PlusArgsTestTop extends Module {
  val io = IO(new Bundle {
    val w = Output(UInt(1.W))
    val x = Output(UInt(1.W))
    val z = Input(UInt(32.W))
  })
  io.w := PlusArgsTest(UInt(32.W), "FOO")
  io.x := PlusArgsTest(io.z, "BAR")
}

class PlusArgsTestSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new PlusArgsTestTop)
    println(fir)
    (fir.split('\n').map(x => x.trim) should contain).allOf(
      "intmodule PlusArgsTestIntrinsic :",
      "output found : UInt<1>",
      "intrinsic = circt.plusargs.test",
      "parameter FORMAT = \"FOO\"",
      "parameter FORMAT = \"BAR\""
    )
  }
}
