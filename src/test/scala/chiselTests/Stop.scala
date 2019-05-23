// See LICENSE for license details.

package chiselTests

import tags.TagRequiresSimulator
import chisel3._
import chisel3.testers.BasicTester

class StopTester() extends BasicTester {
  stop()
}

class StopImmediatelyTester extends BasicTester {
  val cycle = RegInit(0.asUInt(4.W))
  cycle := cycle + 1.U
  when (cycle === 4.U) {
    stop()
  }
  assert(cycle =/= 5.U, "Simulation did not exit upon executing stop()")
}

class StopSpec extends ChiselFlatSpec {
  "stop()" should "stop and succeed the testbench" taggedAs (TagRequiresSimulator) in {
    assertTesterPasses { new StopTester }
  }

  it should "end the simulation immediately" taggedAs (TagRequiresSimulator) in {
    assertTesterPasses { new StopImmediatelyTester }
  }
}
