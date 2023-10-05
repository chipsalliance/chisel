// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.IntrinsicModule
import chisel3.internal.Builder

import circt.Intrinsic

/** A clock gate intrinsic.
  */
private class ClockGateIntrinsic extends IntrinsicModule("circt_clock_gate") {
  val in = IO(Input(Clock()))
  val en = IO(Input(Bool()))
  val out = IO(Output(Clock()))
}

object ClockGate {

  /** Creates an intrinsic which enables and disables a clock safely, without
    * glitches, based on a boolean enable value. If the enable input is 1, the
    * output clock produced by the clock gate is identical to the input clock.
    * If the enable input is 0, the output clock is a constant zero.
    *
    * @example {{{
    * gateClock := ClockGate(clock, enable)
    * }}}
    */
  def apply(input: Clock, enable: Bool): Clock = {
    val inst = Module(new ClockGateIntrinsic)
    inst.in := input
    inst.en := enable
    inst.out
  }
}
