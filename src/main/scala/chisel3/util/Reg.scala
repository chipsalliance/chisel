// See LICENSE for license details.

/** Variations and helpers for registers.
  */

package chisel3.util

import chisel3._

object RegNext {

  def apply[T <: Data](next: T): T = Reg[T](null.asInstanceOf[T], next, null.asInstanceOf[T])

  def apply[T <: Data](next: T, init: T): T = Reg[T](null.asInstanceOf[T], next, init)

}

object RegInit {

  def apply[T <: Data](init: T): T = Reg[T](null.asInstanceOf[T], null.asInstanceOf[T], init)

}

/** A register with an Enable signal */
object RegEnable
{
  def apply[T <: Data](updateData: T, enable: Bool): T = {
    val clonedUpdateData = updateData.chiselCloneType
    val r = Reg(clonedUpdateData)
    when (enable) { r := updateData }
    r
  }
  def apply[T <: Data](updateData: T, resetData: T, enable: Bool): T = {
    val r = RegInit(resetData)
    when (enable) { r := updateData }
    r
  }
}

/** Returns the n-cycle delayed version of the input signal.
  */
object ShiftRegister
{
  /** @param in input to delay
    * @param n number of cycles to delay
    * @param en enable the shift */
  def apply[T <: Data](in: T, n: Int, en: Bool = Bool(true)): T =
  {
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
