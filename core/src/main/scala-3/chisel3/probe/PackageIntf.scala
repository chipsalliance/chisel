// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.experimental.SourceInfo

// Empty olyfill for trait needed by Scala 2
private[chisel3] trait Probe$Intf

/** Access the value of a probe.
  *
  * @param source probe whose value is getting accessed
  */
def read[T <: Data](using SourceInfo)(source: T): T = _readImpl(source)
