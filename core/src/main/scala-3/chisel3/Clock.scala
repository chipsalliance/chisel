// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

object Clock {
  def apply(): Clock = new Clock
}

// TODO: Document this.
sealed class Clock extends ClockImpl {

  /** Returns the contents of the clock wire as a [[Bool]]. */
  def asBool(using SourceInfo): Bool = _asBoolImpl
}
