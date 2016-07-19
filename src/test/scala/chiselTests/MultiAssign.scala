// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class LastAssignTester() extends BasicTester {
  val cnt = Counter(2)

  val test = Wire(UInt(width=4))
  assert(test === 7.U)  // allow read references before assign references

  test := 13.U
  assert(test === 7.U)  // output value should be position-independent

  test := 7.U
  assert(test === 7.U)  // this obviously should work

  when(cnt.value === 1.U) {
    stop()
  }
}

class ReassignmentTester() extends BasicTester {
  val test = 15.U
  test := 7.U
}

class MultiAssignSpec extends ChiselFlatSpec {
  "The last assignment" should "be used when multiple assignments happen" in {
    assertTesterPasses{ new LastAssignTester }
  }
}

class IllegalAssignSpec extends ChiselFlatSpec {
  "Reassignments to non-wire types" should "be disallowed" in {
    intercept[chisel3.internal.ChiselException] {
      assertTesterFails{ new ReassignmentTester }
    }
  }
}
