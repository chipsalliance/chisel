// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.NotStrict.CompileOptions

object RegNext {
  /** Returns a register with the specified next and no reset initialization.
    *
    * Essentially a 1-cycle delayed version of the input signal.
    */
  def apply[T <: Data](next: T): T = Reg[T](null.asInstanceOf[T], next, null.asInstanceOf[T])

  /** Returns a register with the specified next and reset initialization.
    *
    * Essentially a 1-cycle delayed version of the input signal.
    */
  def apply[T <: Data](next: T, init: T): T = Reg[T](null.asInstanceOf[T], next, init)
}

object RegInit {
  /** Returns a register pre-initialized (on reset) to the specified value.
    */
  def apply[T <: Data](init: T): T = Reg[T](null.asInstanceOf[T], null.asInstanceOf[T], init)
}

object RegEnable {
  /** Returns a register with the specified next, update enable gate, and no reset initialization.
    */
  def apply[T <: Data](updateData: T, enable: Bool): T = {
    val clonedUpdateData = updateData.chiselCloneType
    val r = Reg(clonedUpdateData)
    when (enable) { r := updateData }
    r
  }

  /** Returns a register with the specified next, update enable gate, and reset initialization.
    */
  def apply[T <: Data](updateData: T, resetData: T, enable: Bool): T = {
    val r = RegInit(resetData)
    when (enable) { r := updateData }
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
  def apply[T <: Data](in: T, n: Int, en: Bool = Bool(true)): T = {
    // The order of tests reflects the expected use cases.
    if (n == 1) {
      RegEnable(in, en)
    } else if (n != 0) {
      RegNext(apply(in, n-1, en))
    } else {
      in
    }
  }
}
