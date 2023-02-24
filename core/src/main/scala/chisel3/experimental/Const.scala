// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.Builder

/** Create a constant type in FIRRTL, which is guaranteed to take a single
  * constant value.
  */
object Const {
  def apply[T <: Data](source: => T): T = {
    val prevId = Builder.idGen.value
    val data = source // should only evaluate source once
    requireIsChiselType(data)
    val ret = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    ret.isConst = true
    ret
  }
}
