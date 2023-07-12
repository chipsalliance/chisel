// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.Data.ProbeInfo
import chisel3.experimental.SourceInfo
import chisel3.reflect.DataMirror.{checkTypeEquivalence, hasProbeTypeModifier}

import scala.language.experimental.macros

package object probe extends SourceInfoDoc {

  private[chisel3] def setProbeModifier[T <: Data](data: T, probeInfo: Option[ProbeInfo]): Unit = {
    probeInfo.foreach { _ =>
      data.probeInfo = probeInfo
      data match {
        case a: Aggregate =>
          a.elementsIterator.foreach { e => setProbeModifier(e, probeInfo) }
        case _ => // do nothing
      }
    }
  }

  /** Initialize a probe with a provided probe value. */
  def define[T <: Data](sink: T, probeExpr: T)(implicit sourceInfo: SourceInfo): Unit = {
    if (!checkTypeEquivalence(sink, probeExpr)) {
      Builder.error("Cannot define a probe on a non-equivalent type.")
    }
    requireHasProbeTypeModifier(sink, "Expected sink to be a probe.")
    requireHasProbeTypeModifier(probeExpr, "Expected source to be a probe expression.")
    if (sink.probeInfo.get.writable) {
      requireHasWritableProbeTypeModifier(
        probeExpr,
        "Cannot use a non-writable probe expression to define a writable probe."
      )
    }
    pushCommand(ProbeDefine(sourceInfo, sink.ref, probeExpr.ref))
  }

  /** Access the value of a probe. */
  def read[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceRead[T]

  /** @group SourceInfoTransformMacro */
  def do_read[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = {
    requireIsHardware(source)
    requireHasProbeTypeModifier(source)
    // construct clone to bind to ProbeRead
    val clone = source.cloneTypeFull
    clone.bind(OpBinding(Builder.forcedUserModule, Builder.currentWhen))
    clone.setRef(ProbeRead(source.ref))
    // return a non-probe type Data that can be used in Data connects
    clearProbeInfo(clone)
    clone
  }

  /** Recursively clear ProbeInfo */
  private def clearProbeInfo[T <: Data](data: T): Unit = {
    data match {
      case a: Aggregate => {
        a.probeInfo = None
        a.elementsIterator.foreach(x => clearProbeInfo(x))
      }
      case leaf => { leaf.probeInfo = None }
    }
  }

  /** Pad Data if it supports padding */
  private def padData[T <: Data](data: T, width: Int)(implicit sourceInfo: SourceInfo): T = {
    data.widthOption match {
      case None => Builder.error("Data width unknown.")
      case Some(w) =>
        if (w > width) Builder.error(s"Data width $w is larger than $width.")
    }
    data match {
      case d: Bits => d.pad(width).asInstanceOf[T]
      case d => d
    }
  }

  /** Override existing driver of a writable probe on initialization. */
  def forceInitial(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot forceInitial a non-writable Probe.")
    pushCommand(ProbeForceInitial(sourceInfo, probe.ref, padData(value, probe.width.get).ref))
  }

  /** Release initial driver on a probe. */
  def releaseInitial(probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot releaseInitial a non-writable Probe.")
    pushCommand(ProbeReleaseInitial(sourceInfo, probe.ref))
  }

  /** Override existing driver of a writable probe. */
  def force(clock: Clock, cond: Bool, probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot force a non-writable Probe.")
    pushCommand(ProbeForce(sourceInfo, clock.ref, cond.ref, probe.ref, padData(value, probe.width.get).ref))
  }

  /** Release driver on a probe. */
  def release(clock: Clock, cond: Bool, probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot release a non-writable Probe.")
    pushCommand(ProbeRelease(sourceInfo, clock.ref, cond.ref, probe.ref))
  }

}
