// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform

object Clock {
  def apply(): Clock = new Clock
}

// TODO: Document this.
sealed class Clock extends ClockImpl {

  /** Returns the contents of the clock wire as a [[Bool]]. */
  final def asBool: Bool = macro SourceInfoTransform.noArg

  def do_asBool(implicit sourceInfo: SourceInfo): Bool = _asBoolImpl
}
