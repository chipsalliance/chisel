// SPDX-License-Identifier: Apache-2.0
package chisel3.util

import chisel3._

import chisel3.experimental.SourceInfo

private[chisel3] trait CatImpl {

  protected def _applyImpl[T <: Bits](a: T, r: T*)(implicit sourceInfo: SourceInfo): UInt =
    _applyImpl(a :: r.toList)

  protected def _applyImpl[T <: Bits](r: Seq[T])(implicit sourceInfo: SourceInfo): UInt =
    SeqUtils.asUInt(r.reverse)
}
