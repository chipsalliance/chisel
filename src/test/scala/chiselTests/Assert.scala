// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class FailingAssertTester() extends BasicTester {
  assert(false.B)
  // Wait to come out of reset
  val (_, done) = Counter(!reset, 4)
  when (done) {
    stop()
  }
}

class SucceedingAssertTester() extends BasicTester {
  assert(true.B)
  // Wait to come out of reset
  val (_, done) = Counter(!reset, 4)
  when (done) {
    stop()
  }
}

class PipelinedResetModule extends Module {
  val io = IO(new Bundle { })
  val a = Reg(init = 0xbeef.U)
  val b = Reg(init = 0xbeef.U)
  assert(a === b)
}

// This relies on reset being asserted for 3 or more cycles
class PipelinedResetTester extends BasicTester {
  val module = Module(new PipelinedResetModule)

  module.reset := Reg(next = Reg(next = Reg(next = reset)))

  val (_, done) = Counter(!reset, 4)
  when (done) {
    stop()
  }
}

class AssertSpec extends ChiselFlatSpec {
  "A failing assertion" should "fail the testbench" in {
    assert(!runTester{ new FailingAssertTester })
  }
  "A succeeding assertion" should "not fail the testbench" in {
    assertTesterPasses{ new SucceedingAssertTester }
  }
  "An assertion" should "not assert until we come out of reset" in {
    assertTesterPasses{ new PipelinedResetTester }
  }
}
