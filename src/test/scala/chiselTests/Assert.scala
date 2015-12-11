// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class FailingAssertTester() extends BasicTester {
  assert(Bool(false))
  io.done := Bool(true)
  io.error := Bool(false)
}

class SucceedingAssertTester() extends BasicTester {
  assert(Bool(true))
  io.done := Bool(true)
  io.error := Bool(false)
}

class AssertSpec extends ChiselFlatSpec {
  "A failing assertion" should "fail the testbench" in {
    assert(!execute{ new FailingAssertTester })
  }
  "A succeeding assertion" should "not fail the testbench" in {
    assert(execute{ new SucceedingAssertTester })
  }
}
