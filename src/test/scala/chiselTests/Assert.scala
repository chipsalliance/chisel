// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class FailingAssertTester() extends BasicTester {
  assert(Bool(false))
  stop()
}

class SucceedingAssertTester() extends BasicTester {
  assert(Bool(true))
  stop()
}

class AssertSpec extends ChiselFlatSpec {
  "A failing assertion" should "fail the testbench" in {
    assert(!execute{ new FailingAssertTester })
  }
  "A succeeding assertion" should "not fail the testbench" in {
    assert(execute{ new SucceedingAssertTester })
  }
}
