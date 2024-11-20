// SPDX-License-Identifier: Apache-2.0

/** Conditional blocks.
  */

package chisel3.util

import chisel3._

object switch {
  def apply[T <: Element](cond: T)(x: => Any): Unit = println("Not supported yet")
}
