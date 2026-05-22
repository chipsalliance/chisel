// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import chisel3.experimental.SourceInfo

private[chisel3] trait LogicalEach {
  protected def _applyImpl[T <: Data](
    cond: Bool,
    data: T
  )(
    implicit sourceInfo: SourceInfo
  ): T
}

object AndEach extends LogicalEach with LogicalEachIntf {
  protected def _applyImpl[T <: Data](
    cond: Bool,
    data: T
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    Mux(cond, data, (0.U).asTypeOf(data))
  }
}

// Can implement OrEach / XorEach later, if there is a use case
