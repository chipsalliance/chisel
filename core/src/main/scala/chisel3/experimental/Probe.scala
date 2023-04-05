// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.Data.ProbeInfo
import chisel3.internal.Builder
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._

// import firrtl.ir._

/** Utilities for creating and working with probe types.
  */
object Probe {

  /** Create a probe type from a Chisel type. This is not used for
    * hardware.
    */
  private def apply_impl[T <: Data](source: => T, writable: Boolean): T = {
    val prevId = Builder.idGen.value
    val data = source // should only evaluate source once
    requireIsChiselType(data)
    val ret = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    ret.probeInfo = ProbeInfo(true, writable)
    ret
  }

  def apply[T <: Data](source: => T): T = apply_impl(source, false)

  def writable[T <: Data](source: => T): T = apply_impl(source, true)

  def requireIsProbe(probeExpr: Data): Unit = {
    require(probeExpr.probeInfo.isProbe, s"expected $probeExpr to be a probe.")
  }

  def define(sink: Data, probeExpr: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(probeExpr)
    pushCommand(ProbeDefine(sourceInfo, sink.ref, probeExpr.ref))
  }

  def probe[T <: Data](source: => T): Data = {
    val prevId = Builder.idGen.value
    val t = source

    // construct probe to return with cloned info
    val clone = if (!t.mustClone(prevId)) t else t.cloneTypeFull
    clone.bind(chisel3.internal.ProbeBinding(Builder.forcedUserModule, Builder.currentWhen, t))
    clone.setRef(ProbeExpr(t.getRef))
    clone.probeInfo = ProbeInfo(true, t.probeInfo.writable)

    clone
  }

  def read[T <: Data](source: => T): Data = {
    val prevId = Builder.idGen.value
    val t = source
    requireIsProbe(t)

    // construct probe to return with cloned info
    val clone = if (!t.mustClone(prevId)) t else t.cloneTypeFull
    clone.bind(chisel3.internal.ProbeBinding(Builder.forcedUserModule, Builder.currentWhen, t))
    clone.setRef(ProbeRead(t.getRef))

    clone
  }

  def forceInitial(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(probe)
    pushCommand(ProbeForceInitial(sourceInfo, probe.ref, value.ref))
  }

  def releaseInitial(probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(probe)
    pushCommand(ProbeReleaseInitial(sourceInfo, probe.ref))
  }

  def force(clock: Clock, cond: Data, probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(probe)
    pushCommand(ProbeForce(sourceInfo, clock.ref, cond.ref, probe.ref, value.ref))
  }

  def release(clock: Clock, cond: Data, probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(probe)
    pushCommand(ProbeRelease(sourceInfo, clock.ref, cond.ref, probe.ref))
  }

}
