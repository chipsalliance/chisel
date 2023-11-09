// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.circt

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.circt.ClockGate
import circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

private class ClockGateTop extends RawModule {
  val clock = IO(Input(Clock()))
  val enable = IO(Input(Bool()))
  val gatedClock = IO(Output(Clock()))
  gatedClock := ClockGate(clock, enable)
}

class ClockGateSpec extends AnyFlatSpec with Matchers {
  it should "gate clocks" in {
    val fir = ChiselStage.emitCHIRRTL(new ClockGateTop)
    (fir.split('\n').map(_.trim.takeWhile(_ != '@')) should contain).allOf(
      "intmodule ClockGateIntrinsic : ",
      "input in : Clock",
      "input en : UInt<1>",
      "output out : Clock",
      "intrinsic = circt_clock_gate"
    )
  }
}
