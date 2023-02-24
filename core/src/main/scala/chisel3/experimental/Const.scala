// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._

/** Create a constant type in FIRRTL, which is guaranteed to take a single
  * constant value.
  */
object Const {
  def apply[A <: Data](a: A): A = {
    a.isConst = true
    a
  }
}
