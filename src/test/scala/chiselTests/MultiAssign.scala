// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class LastAssignTester() extends BasicTester {
  val countOnClockCycles = true.B
  val (cnt, wrap) = Counter(countOnClockCycles,2)

  val test = Wire(UInt(4.W))
  assert(test === 7.U)  // allow read references before assign references

  test := 13.U
  assert(test === 7.U)  // output value should be position-independent

  test := 7.U
  assert(test === 7.U)  // this obviously should work

  when(cnt === 1.U) {
    stop()
  }
}

class MultiAssignSpec extends ChiselFlatSpec {
  "The last assignment" should "be used when multiple assignments happen" in {
    assertTesterPasses{ new LastAssignTester }
  }
}

class IllegalAssignSpec extends ChiselFlatSpec {
  "Reassignments to literals" should "be disallowed" in {
    intercept[chisel3.internal.ChiselException] {
      elaborate{ new BasicTester {
        15.U := 7.U
      }}
    }
  }
  
  "Reassignments to ops" should "be disallowed" in {
    intercept[chisel3.internal.ChiselException] {
      elaborate{ new BasicTester {
        (15.U + 1.U) := 7.U
      }}
    }
  }
  
  "Reassignments to bit slices" should "be disallowed" in {
    intercept[chisel3.internal.ChiselException] {
      elaborate{ new BasicTester {
        (15.U)(1, 0) := 7.U
      }}
    }
  }
  
  "Bulk-connecting two read-only nodes" should "be disallowed" in {
    intercept[chisel3.internal.ChiselException] {
      elaborate{ new BasicTester {
        (15.U + 1.U) <> 7.U
      }}
    }
  }
}
