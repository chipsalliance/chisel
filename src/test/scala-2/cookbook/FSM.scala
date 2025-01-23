// SPDX-License-Identifier: Apache-2.0

package cookbook

import chisel3._
import chisel3.util._

/* ### How do I create a finite state machine?
 *
 * Use ChiselEnum to construct the states and switch & is to construct the FSM
 * control logic
 */

class DetectTwoOnes extends Module {
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
  })

  object State extends ChiselEnum {
    val sNone, sOne1, sTwo1s = Value
  }

  val state = RegInit(State.sNone)

  io.out := (state === State.sTwo1s)

  switch(state) {
    is(State.sNone) {
      when(io.in) {
        state := State.sOne1
      }
    }
    is(State.sOne1) {
      when(io.in) {
        state := State.sTwo1s
      }.otherwise {
        state := State.sNone
      }
    }
    is(State.sTwo1s) {
      when(!io.in) {
        state := State.sNone
      }
    }
  }
}

class DetectTwoOnesTester extends CookbookTester(10) {

  val dut = Module(new DetectTwoOnes)

  // Inputs and expected results
  val inputs: Vec[Bool] = VecInit(false.B, true.B, false.B, true.B, true.B, true.B, false.B, true.B, true.B, false.B)
  val expected: Vec[Bool] =
    VecInit(false.B, false.B, false.B, false.B, false.B, true.B, true.B, false.B, false.B, true.B)

  dut.io.in := inputs(cycle)
  assert(dut.io.out === expected(cycle))
}

class FSMSpec extends CookbookSpec {
  "DetectTwoOnes" should "work" in {
    assertTesterPasses { new DetectTwoOnesTester }
  }
}
