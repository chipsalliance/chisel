// See LICENSE for license details.

package chiselTests

import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

class LastAssignTester() extends BasicTester {
  val cnt = Counter(2)

  val test = Wire(UInt(width=4))
  assert(test === UInt(7))  // allow read references before assign references

  test := UInt(13)
  assert(test === UInt(7))  // output value should be position-independent

  test := UInt(7)
  assert(test === UInt(7))  // this obviously should work

  when(cnt.value === UInt(1)) {
    stop()
  }
}

class ReassignmentTester() extends BasicTester {
  val test = UInt(15)
  test := UInt(7)
}

class MultiAssignSpec extends ChiselFlatSpec {
  "The last assignment" should "be used when multiple assignments happen" in {
    assert(execute{ new LastAssignTester })
  }
  "Reassignments to non-wire types" should "be disallowed" in {
    assert(!execute{ new ReassignmentTester })
  }
}
