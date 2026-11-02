// SPDX-License-Identifier: Apache-2.0

package chiselTests.chiseltest

import chisel3._
import chisel3.ltl.AssertProperty
import chisel3.util._
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec

/** Formal harness for the combinational, fixed-priority Arbiter.
  * All input valid bits, payloads, and downstream ready are unconstrained.
  */
class ArbiterFormalHarness(n: Int, width: Int) extends Module {
  val io = IO(new ArbiterIO(UInt(width.W), n))
  val dut = Module(new Arbiter(UInt(width.W), n))
  io :<>= dut.io

  val anyValid = io.in.map(_.valid).reduce(_ || _)
  AssertProperty(io.out.valid === anyValid, "output_valid_iff_any_input_valid")
  AssertProperty(io.chosen < n.U, "chosen_in_range")

  // With no requests, Arbiter defaults to the last input (including its payload).
  AssertProperty(anyValid || io.chosen === (n - 1).U, "idle_choice_is_last_input")

  for (i <- 0 until n) {
    val higherPriorityValid = io.in.take(i).map(_.valid).foldLeft(false.B)(_ || _)
    val selected = io.in(i).valid && !higherPriorityValid

    AssertProperty(!selected || io.chosen === i.U, s"lowest_valid_input_${i}_wins")
    AssertProperty(io.chosen =/= i.U || io.out.bits === io.in(i).bits, s"input_${i}_payload_routing")

    // Ready does not depend on this input's own valid bit. When out.ready is
    // high, every input up to and including the first valid one is ready, but
    // only that first valid input can transfer.
    AssertProperty(
      io.in(i).ready === (io.out.ready && !higherPriorityValid),
      s"input_${i}_ready_and_backpressure"
    )
    // Derived: implied by the ready, valid, and chosen properties above. Kept
    // to document the handshake contract.
    AssertProperty(
      io.in(i).fire === (io.out.fire && io.chosen === i.U),
      s"input_${i}_transfer_matches_output"
    )
  }

  // Exactly one input transfers on each output transfer, and none otherwise.
  // Derived: implied by the per-input transfer properties above. Kept to
  // document the handshake contract.
  AssertProperty(PopCount(io.in.map(_.fire)) === io.out.fire.asUInt, "transfer_conservation")
}

class ArbiterFormalSpec extends AnyFlatSpec with ChiselScalatestTester with Formal {
  behavior of "Arbiter"

  // Include the zero-width chosen index (n = 1), powers of two, and counts
  // with unused chosen encodings. Payload widths cover the zero-width edge
  // case plus a narrow and a wide payload; routing is a plain equality, so
  // more intermediate widths add no coverage.
  for {
    n <- Seq(1, 2, 3, 4, 5, 8)
    width <- Seq(0, 1, 8)
  } {
    it should s"satisfy fixed-priority arbitration with $n inputs and $width-bit payloads" in {
      // Arbiter is purely combinational, so we don't need more than 1 cycle of bound, but 2 gives
      // margin.
      val res = verifyRes(
        new ArbiterFormalHarness(n, width),
        Seq(
          BoundedCheck(2),
          BTORMCBackend
        )
      )

      assertResult(Unsat)(res)
    }
  }
}
