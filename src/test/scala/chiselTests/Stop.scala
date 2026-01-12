// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class StopTester() extends Module {
  chisel3.stop()
}

class StopWithMessageTester() extends Module {
  val cycle = RegInit(0.U(4.W))
  cycle := cycle + 1.U
  when(cycle === 4.U) {
    chisel3.stop(cf"cycle: $cycle")
  }
}

class StopImmediatelyTester extends Module {
  val cycle = RegInit(0.asUInt(4.W))
  cycle := cycle + 1.U
  when(cycle === 4.U) {
    chisel3.stop()
  }
  assert(cycle =/= 5.U, "Simulation did not exit upon executing stop()")
}

class StopSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "stop()" should "stop and succeed the testbench" in {
    simulate { new StopTester }(RunUntilFinished(3))
  }

  it should "emit an optional message with arguments" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new StopWithMessageTester)
    chirrtl should include("\"cycle: %d\", cycle")
  }

  it should "end the simulation immediately" in {
    simulate { new StopImmediatelyTester }(RunUntilFinished(6))
  }
}
