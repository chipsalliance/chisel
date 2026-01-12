// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.Counter
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LastAssignTester() extends Module {
  val countOnClockCycles = true.B
  val (cnt, wrap) = Counter(countOnClockCycles, 2)

  val test = Wire(UInt(4.W))
  assert(test === 7.U) // allow read references before assign references

  test := 13.U
  assert(test === 7.U) // output value should be position-independent

  test := 7.U
  assert(test === 7.U) // this obviously should work

  when(cnt === 1.U) {
    stop()
  }
}

class MultiAssignSpec extends AnyFlatSpec with Matchers with ChiselSim {
  "The last assignment" should "be used when multiple assignments happen" in {
    simulate { new LastAssignTester }(RunUntilFinished(3))
  }
}

class IllegalAssignSpec extends AnyFlatSpec with Matchers {
  "Reassignments to literals" should "be disallowed" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          15.U := 7.U
        }
      }
    }
  }

  "Reassignments to ops" should "be disallowed" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          (15.U + 1.U) := 7.U
        }
      }
    }
  }

  "Reassignments to bit slices" should "be disallowed" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          (15.U)(1, 0) := 7.U
        }
      }
    }
  }

  "Bulk-connecting two read-only nodes" should "be disallowed" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          (15.U + 1.U) <> 7.U
        }
      }
    }
  }
}
