// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.experimental.SourceInfo

import scala.language.experimental.macros

package object probe {

  /** Initialize a probe with a provided probe value. */
  def define(sink: Data, probeExpr: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasProbeTypeModifier(sink)
    requireHasProbeTypeModifier(probeExpr)
    pushCommand(ProbeDefine(sourceInfo, sink.ref, probeExpr.ref))
  }

  /** Access the value of a probe. */
  def read[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceRead[T]

  def do_read[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = {
    // construct probe to return with cloned info
    val clone = source.cloneTypeFull
    clone.bind(chisel3.internal.ProbeBinding(Builder.forcedUserModule, Builder.currentWhen, source))
    clone.setRef(ProbeRead(source.ref))
    clone.probeInfo = source.probeInfo
    clone
  }

  /** Override existing driver of a writable probe on initialization. */
  def forceInitial(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe)
    pushCommand(ProbeForceInitial(sourceInfo, probe.ref, value.ref))
  }

  /** Release initial driver on a probe. */
  def releaseInitial(probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe)
    pushCommand(ProbeReleaseInitial(sourceInfo, probe.ref))
  }

  /** Override existing driver of a writable probe. */
  def force(clock: Clock, cond: Bool, probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe)
    pushCommand(ProbeForce(sourceInfo, clock.ref, cond.ref, probe.ref, value.ref))
  }

  /** Release driver on a probe. */
  def release(clock: Clock, cond: Bool, probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe)
    pushCommand(ProbeRelease(sourceInfo, clock.ref, cond.ref, probe.ref))
  }

}
