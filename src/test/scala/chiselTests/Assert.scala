// See LICENSE for license details.

package chiselTests

import org.scalatest._
import chisel._
import chisel.testers.BasicTester
import chisel.util._

class FailingAssertTester() extends BasicTester {
  assert(Bool(false))
  // Wait to come out of reset
  val (_, done) = Counter(!reset, 4)
  when (done) {
    stop()
  }
}

class SucceedingAssertTester() extends BasicTester {
  assert(Bool(true))
  // Wait to come out of reset
  val (_, done) = Counter(!reset, 4)
  when (done) {
    stop()
  }
}

class PipelinedResetModule extends Module {
  val io = new Bundle { }
  val a = Reg(init = UInt(0xbeef))
  val b = Reg(init = UInt(0xbeef))
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
