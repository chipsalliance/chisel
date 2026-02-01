// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait SwitchContext$Intf[T <: Element] { self: SwitchContext[T] =>

  def is(
    v: Iterable[T]
  )(block: => Any)(
    implicit sourceInfo: SourceInfo
  ): SwitchContext[T] = _isImpl(v)(block)

  def is(v: T)(block: => Any)(implicit sourceInfo: SourceInfo): SwitchContext[T] = _isImpl(v)(block)

  def is(
    v:  T,
    vr: T*
  )(block: => Any)(
    implicit sourceInfo: SourceInfo
  ): SwitchContext[T] = _isImpl(v, vr: _*)(block)
}
