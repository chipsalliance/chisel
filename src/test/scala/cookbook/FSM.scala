// See LICENSE for license details.

package cookbook

import chisel3._
import chisel3.util._

/* ### How do I create a finite state machine?
 *
 * Use Chisel Enum to construct the states and switch & is to construct the FSM
 * control logic
 */
class DetectTwoOnes extends Module {
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
  })

  val sNone :: sOne1 :: sTwo1s :: Nil = Enum(3)
  val state = RegInit(sNone)

  io.out := (state === sTwo1s)

  switch (state) {
    is (sNone) {
      when (io.in) {
        state := sOne1
      }
    }
    is (sOne1) {
      when (io.in) {
        state := sTwo1s
      } .otherwise {
        state := sNone
      }
    }
    is (sTwo1s) {
      when (!io.in) {
        state := sNone
      }
    }
  }
}

class DetectTwoOnesTester extends CookbookTester(10) {

  val dut = Module(new DetectTwoOnes)

  // Inputs and expected results
  val inputs: Vec[Bool] = VecInit(false.B, true.B, false.B, true.B, true.B, true.B, false.B, true.B, true.B, false.B)
  val expected: Vec[Bool] = VecInit(false.B, false.B, false.B, false.B, false.B, true.B, true.B, false.B, false.B, true.B)

  dut.io.in := inputs(cycle)
  assert(dut.io.out === expected(cycle))
}

class FSMSpec extends CookbookSpec {
  "DetectTwoOnes" should "work" in {
    assertTesterPasses { new DetectTwoOnesTester }
  }
}
