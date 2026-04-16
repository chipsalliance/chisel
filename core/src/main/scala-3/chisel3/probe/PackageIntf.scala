// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.experimental.SourceInfo

// Empty polyfill for trait needed by Scala 2
private[chisel3] trait ProbeObjIntf

/** Access the value of a probe.
  *
  * @param source probe whose value is getting accessed
  */
def read[T <: Data](source: T)(using SourceInfo): T = _readImpl(source)
