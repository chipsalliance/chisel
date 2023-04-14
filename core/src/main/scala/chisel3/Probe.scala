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
private[chisel3] sealed trait ProbeBase {

  private def requireIsProbe(probeExpr: Data)(implicit sourceInfo: SourceInfo): Unit = {
    if (!isProbe(probeExpr)) Builder.error(s"expected $probeExpr to be a probe.")
  }

  private def requireIsWritableProbe(probeExpr: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(probeExpr)
    if (!probeExpr.probeInfo.get.writable) Builder.error(s"expected $probeExpr to be writable.")
  }

  protected def apply[T <: Data](source: => T, writable: Boolean): T = {
    val prevId = Builder.idGen.value
    // call Output() to coerce passivity
    val data = Output(source) // should only evaluate source once
    requireIsChiselType(data)
    val ret = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    ret.probeInfo = Some(ProbeInfo(writable))
    ret
  }

  /** Initialize a Probe with a provided probe expression. */
  def define(sink: Data, probeExpr: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireIsProbe(sink)
    requireIsProbe(probeExpr)
    pushCommand(ProbeDefine(sourceInfo, sink.ref, probeExpr.ref))
  }

  /** Access the value of a probe. */
  def read[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = {
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

object Probe extends ProbeBase {

  /** Create a read-only probe type from a Chisel type. This is only used for
    * ports.
    */
  def apply[T <: Data](source: => T): T = super.apply(source, false)
}

object RWProbe extends ProbeBase {

  /** Create a writable probe type from a Chisel type. This is only used for
    * ports.
    */
  def apply[T <: Data](source: => T): T = super.apply(source, true)
}
