// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.experimental.SourceInfo

import scala.language.experimental.macros

private[chisel3] trait Probe$Intf extends SourceInfoDoc {

  /** Access the value of a probe.
    *
    * @param source probe whose value is getting accessed
    */
  def read[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceRead[T]

  /** @group SourceInfoTransformMacro */
  def do_read[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = probe._readImpl(source)

  /** Initialize a probe with a provided probe value.
    *
    * @param sink probe to initialize
    * @param probeExpr value to initialize the sink to
    */
  def define[T <: Data](sink: T, probeExpr: T)(implicit sourceInfo: SourceInfo): Unit = _defineImpl(sink, probeExpr)

  /** Override existing driver of a writable probe on initialization.
    *
    * @param probe writable Probe to force
    * @param value to force onto the probe
    */
  def forceInitial(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = _forceInitialImpl(probe, value)

  /** Release initial driver on a probe.
    *
    * @param probe writable Probe to release
    */
  def releaseInitial(probe: Data)(implicit sourceInfo: SourceInfo): Unit = _releaseInitialImpl(probe)

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
  def force(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = _forceImpl(probe, value)

  /** Release driver on a probe. If called within the scope of a [[when]]
    * block, the release will only occur on cycles that the when condition
    * is true.
    *
    * Fires only when reset has been asserted and then deasserted through the
    * [[Disable]] API.
    *
    * @param probe writable Probe to release
    */
  def release(probe: Data)(implicit sourceInfo: SourceInfo): Unit = _releaseImpl(probe)
}
