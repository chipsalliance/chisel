// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.circt

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.PlusArgsTest
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class PlusArgsTestTop extends Module {
  val w = IO(Output(UInt(1.W)))
  val x = IO(Output(UInt(1.W)))
  val z = IO(Input(UInt(32.W)))
  w := PlusArgsTest(UInt(32.W), "FOO")
  x := PlusArgsTest(z, "BAR")
}

class PlusArgsTestSpec extends AnyFlatSpec with Matchers {
  it should "work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new PlusArgsTestTop)
    (fir.split('\n').map(_.trim.takeWhile(_ != '@')) should contain).allOf(
      "intmodule PlusArgsTestIntrinsic : ",
      "output found : UInt<1>",
      "intrinsic = circt_plusargs_test",
      "parameter FORMAT = \"FOO\"",
      "parameter FORMAT = \"BAR\""
    )
  }
}
