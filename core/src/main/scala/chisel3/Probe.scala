// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.Data.ProbeInfo
import chisel3.reflect.DataMirror.hasProbeTypeModifier
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.experimental.SourceInfo

import scala.language.experimental.macros

/** Utilities for creating and working with Chisel types that have a probe or
  * writable probe modifier.
  */
private[chisel3] sealed trait ProbeBase {

  protected def apply[T <: Data](source: => T, writable: Boolean)(implicit sourceInfo: SourceInfo): T = {
    val prevId = Builder.idGen.value
    // call Output() to coerce passivity
    val data = Output(source) // should only evaluate source once
    requireIsChiselType(data)
    requireNoProbeTypeModifier(data, "Cannot probe a probe.")
    //require is not aggregate containing probe using datamirror

    val ret = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    ret.probeInfo = Some(ProbeInfo(writable))
    ret
  }

  /** Initialize a probe with a provided probe value. */
  def define(sink: Data, probeExpr: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasProbeTypeModifier(sink)
    requireHasProbeTypeModifier(probeExpr)
    pushCommand(ProbeDefine(sourceInfo, sink.ref, probeExpr.ref))
  }

  /** Access the value of a probe. */
  def read[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceRead[T]

  def do_read[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = {
    val prevId = Builder.idGen.value
    val t = source
    requireHasProbeTypeModifier(t)

    // construct probe to return with cloned info
    val clone = if (!t.mustClone(prevId)) t else t.cloneTypeFull
    clone.bind(chisel3.internal.ProbeBinding(Builder.forcedUserModule, Builder.currentWhen, t))
    clone.setRef(ProbeRead(t.ref))
    clone.probeInfo = t.probeInfo

    clone
  }
}

object Probe extends ProbeBase {

  /** Mark a Chisel type as with a probe modifier.
    */
  def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = super.apply(source, false)
}

object RWProbe extends ProbeBase {

  /** Mark a Chisel type with a writable probe modifier.
    */
  def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = super.apply(source, true)

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
