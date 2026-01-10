// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3.{Data, SourceInfoDoc}
import chisel3.experimental.SourceInfo

object ProbeValue extends ProbeValueBase with SourceInfoDoc {

  /** Create a read-only probe expression. */
  def apply[T <: Data](using SourceInfo)(source: T): T = super.apply(source, writable = false)
}

object RWProbeValue extends ProbeValueBase with SourceInfoDoc {

  /** Create a read/write probe expression. */
  def apply[T <: Data](using SourceInfo)(source: T): T = super.apply(source, writable = true)
}
