// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait Const$Intf { self: Const.type =>

  def apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = _applyImpl(source)
}
