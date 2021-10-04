// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._

object RegEnable {
  /** Returns a register with the specified next, update enable gate, and no reset initialization.
    *
    * @example {{{
    * val regWithEnable = RegEnable(nextVal, ena)
    * }}}
    */
  def apply[T <: Data](next: T, enable: Bool): T = {
    val r = Reg(chiselTypeOf(next))
    when (enable) { r := next }
    r
  }

  /** Returns a register with the specified next, update enable gate, and reset initialization.
    *
    * @example {{{
    * val regWithEnableAndReset = RegEnable(nextVal, 0.U, ena)
    * }}}
    */
  def apply[T <: Data](next: T, init: T, enable: Bool): T = {
    val r = RegInit(init)
    when (enable) { r := next }
    r
  }
}

object ShiftRegister
{
  /** Returns the n-cycle delayed version of the input signal.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    * @param en enable the shift
    *
    * @example {{{
    * val regDelayTwo = ShiftRegister(nextVal, 2, ena)
    * }}}
    */
  def apply[T <: Data](in: T, n: Int, en: Bool = true.B): T = ShiftRegisters(in, n, en).lastOption.getOrElse(in)

  /** Returns the n-cycle delayed version of the input signal with reset initialization.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    * @param resetData reset value for each register in the shift
    * @param en enable the shift
    *
    * @example {{{
    * val regDelayTwoReset = ShiftRegister(nextVal, 2, 0.U, ena)
    * }}}
    */
  def apply[T <: Data](in: T, n: Int, resetData: T, en: Bool): T = ShiftRegisters(in, n, resetData, en).lastOption.getOrElse(in)
}


object ShiftRegisters
{
  /** Returns a sequence of delayed input signal registers from 1 to n.
    *
    * @param in input to delay
    * @param n  number of cycles to delay
    * @param en enable the shift
    *
    */
  def apply[T <: Data](in: T, n: Int, en: Bool = true.B): Seq[T] =
    Seq.iterate(in, n + 1)(util.RegEnable(_, en)).drop(1)

  /** Returns delayed input signal registers with reset initialization from 1 to n.
    *
    * @param in        input to delay
    * @param n         number of cycles to delay
    * @param resetData reset value for each register in the shift
    * @param en        enable the shift
    *
    */
  def apply[T <: Data](in: T, n: Int, resetData: T, en: Bool): Seq[T] =
    Seq.iterate(in, n + 1)(util.RegEnable(_, resetData, en)).drop(1)
}
