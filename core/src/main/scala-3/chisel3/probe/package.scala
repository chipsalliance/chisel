// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.experimental.SourceInfo

package object probe extends ObjectProbeImpl {

  /** Access the value of a probe.
    *
    * @param source probe whose value is getting accessed
    */
  def read[T <: Data](source: T)(using SourceInfo): T = _readImpl(source)
}
