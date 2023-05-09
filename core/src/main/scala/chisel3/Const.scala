// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.internal.{requireIsChiselType, requireNoProbeTypeModifier, Builder}
import chisel3.experimental.SourceInfo

/** Create a constant type in FIRRTL, which is guaranteed to take a single
  * constant value.
  */
object Const {
  def apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = {
    val prevId = Builder.idGen.value
    val data = source // should only evaluate source once
    requireIsChiselType(data)
    requireNoProbeTypeModifier(data, "Cannot create Const of a Probe.")
    val ret = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    ret.isConst = true
    ret
  }
}
