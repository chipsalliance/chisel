// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.Data.ProbeInfo
import chisel3.reflect.DataMirror.isProbe
import chisel3.internal.{requireIsChiselType, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.firrtl._

/** Utilities for creating and working with probe types.
  */
object Probe {

  private def apply_impl[T <: Data](source: => T, writable: Boolean): T = {
    val prevId = Builder.idGen.value
    // call Output() to coerce passivity
    val data = Output(source) // should only evaluate source once
    requireIsChiselType(data)
    val ret = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    ret.probeInfo = Some(ProbeInfo(writable))
    ret
  }

  def _autoProbe[T](a: T)(implicit si: SourceInfo): T = a match {
    case d: Data => Builder.currentModule.get._refsToProbe += ((d, si)); a
    case _ => a
  }

  /** Create a read-only probe type from a Chisel type. This is only used for
    * ports and not for hardware.
    */
  def apply[T <: Data](source: => T): T = apply_impl(source, false)

  /** Create a writable probe type from a Chisel type. This is only used for
    * ports and not for hardware.
    */
  def writable[T <: Data](source: => T): T = apply_impl(source, true)

  private def requireIsProbe(probeExpr: Data): Unit = {
    require(isProbe(probeExpr), s"expected $probeExpr to be a probe.")
  }

  private def requireIsWritableProbe(probeExpr: Data): Unit = {
    requireIsProbe(probeExpr)
    require(probeExpr.probeInfo.get.writable, s"expected $probeExpr to be writable.")
  }

  /** Initialize a Probe with a provided probe expression. */
  def define(sink: Data, probeExpr: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(sink)
    requireIsProbe(probeExpr)
    pushCommand(ProbeDefine(sourceInfo, sink.ref, probeExpr.ref))
  }

  private def probe_impl[T <: Data](source: T, writable: Boolean): Data = {
    // construct probe to return with cloned info
    val clone = Probe.apply_impl(source.cloneType, writable)
    clone.bind(chisel3.internal.ProbeBinding(Builder.forcedUserModule, Builder.currentWhen, source))
    clone.setRef(ProbeExpr(source.ref))

    clone
  }

  /** Create a read-only probe. */
  def probe[T <: Data](source: => T): Data = probe_impl(source, false)

  /** Create a read/write probe. */
  def rwprobe[T <: Data](source: => T): Data = probe_impl(source, true)

  /** Access the value of a probe. */
  def read[T <: Data](source: => T): Data = {
    val prevId = Builder.idGen.value
    val t = source
    requireIsProbe(t)

    // construct probe to return with cloned info
    val clone = if (!t.mustClone(prevId)) t else t.cloneTypeFull
    clone.bind(chisel3.internal.ProbeBinding(Builder.forcedUserModule, Builder.currentWhen, t))
    clone.setRef(ProbeRead(t.ref))
    clone.probeInfo = t.probeInfo

    clone
  }

  /** Override existing driver of a writable probe on initialization. */
  def forceInitial(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsWritableProbe(probe)
    pushCommand(ProbeForceInitial(sourceInfo, probe.ref, value.ref))
  }

  /** Release initial driver on a probe. */
  def releaseInitial(probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsWritableProbe(probe)
    pushCommand(ProbeReleaseInitial(sourceInfo, probe.ref))
  }

  /** Override existing driver of a writable probe. */
  def force(clock: Clock, cond: Bool, probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsWritableProbe(probe)
    pushCommand(ProbeForce(sourceInfo, clock.ref, cond.ref, probe.ref, value.ref))
  }

  /** Release driver on a probe. */
  def release(clock: Clock, cond: Bool, probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsWritableProbe(probe)
    pushCommand(ProbeRelease(sourceInfo, clock.ref, cond.ref, probe.ref))
  }

}
