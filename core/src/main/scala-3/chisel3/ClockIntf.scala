// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait ClockIntf { self: Clock =>

  /** Returns the contents of the clock wire as a [[Bool]]. */
  def asBool(using SourceInfo): Bool = _asBoolImpl
}
