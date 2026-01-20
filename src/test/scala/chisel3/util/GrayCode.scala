// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.ltl.AssertProperty
import chisel3.test.UnitTest

/** Exhaustively test the gray conversion on 16 bits. */
object GrayCodeExhaustive extends RawModule with UnitTest with SimulationTestHarnessInterface with Public {
  // FIXME: `Public` should not be necessary since the module is explicitly
  // marked as a `SimulationTest`. A bug in firtool currently causes ports to
  // be deleted when the modules is not public. Remove `Public` when the bug
  // is fixed.
  val clock = IO(Input(Clock()))
  val init = IO(Input(Bool()))
  val done = IO(Output(Bool()))
  val success = IO(Output(Bool()))
  SimulationTest(this)

  withClockAndReset(clock, init) {
    val value = RegInit(0.U(16.W))
    val numMismatches = RegInit(0.U((value.getWidth + 1).W))
    val gray = BinaryToGray(value)
    val binary = GrayToBinary(gray)
    done := false.B
    when(value.andR || numMismatches >= 10.U) {
      done := true.B
    }.otherwise {
      when(value =/= binary) {
        printf(p"MISMATCH: $value -> $gray -> $binary\n")
        numMismatches := numMismatches + 1.U
      }
      value := value + 1.U
    }
    success := numMismatches === 0.U
  }
}
