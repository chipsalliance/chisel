// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester

class MultiSwitchFireTester extends BasicTester {
  val wire = WireInit(0.U)
  switch (0.U) {
    is (0.U) { wire := 1.U } // Only this one should happen
    is (0.U) { wire := 2.U }
  }
  assert(wire === 1.U)
  stop()
}

class SwitchSpec extends ChiselFlatSpec {
  "switch" should "work for implementing FSMs" in {
    assertTesterPasses(new MultiSwitchFireTester)
  }
}
