// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.experimental.SourceInfo

// Empty polyfill for trait needed by Scala 2
private[chisel3] trait Probe$Intf

/** Access the value of a probe.
  *
  * @param source probe whose value is getting accessed
  */
def read[T <: Data](source: T)(using SourceInfo): T = _readImpl(source)

/** Initialize a probe with a provided probe value.
  *
  * @param sink probe to initialize
  * @param probeExpr value to initialize the sink to
  */
def define[T <: Data](sink: T, probeExpr: T)(using SourceInfo): Unit = _defineImpl(sink, probeExpr)

/** Override existing driver of a writable probe on initialization.
  *
  * @param probe writable Probe to force
  * @param value to force onto the probe
  */
def forceInitial(probe: Data, value: Data)(using SourceInfo): Unit = _forceInitialImpl(probe, value)

/** Release initial driver on a probe.
  *
  * @param probe writable Probe to release
  */
def releaseInitial(probe: Data)(using SourceInfo): Unit = _releaseInitialImpl(probe)

/** Override existing driver of a writable probe. If called within the scope
  * of a [[when]] block, the force will only occur on cycles that the when
  * condition is true.
  *
  * Fires only when reset has been asserted and then deasserted through the
  * [[Disable]] API.
  *
  * @param probe writable Probe to force
  * @param value to force onto the probe
  */
def force(probe: Data, value: Data)(using SourceInfo): Unit = _forceImpl(probe, value)

/** Release driver on a probe. If called within the scope of a [[when]]
  * block, the release will only occur on cycles that the when condition
  * is true.
  *
  * Fires only when reset has been asserted and then deasserted through the
  * [[Disable]] API.
  *
  * @param probe writable Probe to release
  */
def release(probe: Data)(using SourceInfo): Unit = _releaseImpl(probe)
