// See LICENSE for license details.

package chisel3.util

import chisel3._

object RegEnable {
  /** Returns a register with the specified next, update enable gate, and no reset initialization.
    */
  def apply[T <: Data](next: T, enable: Bool): T = {
    val r = Reg(chiselTypeOf(next))
    when (enable) { r := next }
    r
  }

  /** Returns a register with the specified next, update enable gate, and reset initialization.
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
    */
  def apply[T <: Data](in: T, n: Int, en: Bool = true.B): T = {
    // The order of tests reflects the expected use cases.
    if (n != 0) {
      RegEnable(apply(in, n-1, en), en)
    } else {
      in
    }
  }

  /** Returns the n-cycle delayed version of the input signal with reset initialization.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    * @param resetData reset value for each register in the shift
    * @param en enable the shift
    */
  def apply[T <: Data](in: T, n: Int, resetData: T, en: Bool): T = {
    // The order of tests reflects the expected use cases.
    if (n != 0) {
      RegEnable(apply(in, n-1, resetData, en), resetData, en)
    } else {
      in
    }
  }
}
