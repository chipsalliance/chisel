// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import _root_.circt.stage.ChiselStage

class StopTester() extends BasicTester {
  chisel3.stop()
}

class StopWithMessageTester() extends BasicTester {
  val cycle = RegInit(0.U(4.W))
  cycle := cycle + 1.U
  when(cycle === 4.U) {
    chisel3.stop(cf"cycle: $cycle")
  }
}

class StopImmediatelyTester extends BasicTester {
  val cycle = RegInit(0.asUInt(4.W))
  cycle := cycle + 1.U
  when(cycle === 4.U) {
    chisel3.stop()
  }
  assert(cycle =/= 5.U, "Simulation did not exit upon executing stop()")
}

class StopSpec extends ChiselFlatSpec {
  "stop()" should "stop and succeed the testbench" in {
    assertTesterPasses { new StopTester }
  }

  it should "emit an optional message with arguments" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new StopWithMessageTester)
    chirrtl should include("\"cycle: %d\", cycle")
  }

  it should "end the simulation immediately" in {
    assertTesterPasses { new StopImmediatelyTester }
  }
}
