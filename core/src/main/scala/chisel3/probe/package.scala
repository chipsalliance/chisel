// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.Data.ProbeInfo
import chisel3.experimental.SourceInfo
import chisel3.reflect.DataMirror.{checkTypeEquivalence, collectAllMembers, hasProbeTypeModifier}

import scala.language.experimental.macros

package object probe extends SourceInfoDoc {

  private[chisel3] def setProbeModifier[T <: Data](data: T, probeInfo: Option[ProbeInfo]): Unit = {
    probeInfo.foreach { _ =>
      collectAllMembers(data).foreach { member =>
        member.probeInfo = probeInfo
        // also set sample_element's probe information in Vecs
        member match {
          case v: Vec[_] => v.sample_element.probeInfo = probeInfo
          case _ => // do nothing
        }
      }
    }
  }

  /** Initialize a probe with a provided probe value.
    *
    * @param sink probe to initialize
    * @param probeExpr value to initialize the sink to
    */
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

  /** Access the value of a probe.
    *
    * @param source probe whose value is getting accessed
    */
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

  /** Pad [[Data]] to the width of a probe if it supports padding */
  private def padDataToProbeWidth[T <: Data](data: T, probe: Data)(implicit sourceInfo: SourceInfo): T = {
    // check probe width is known
    requireHasProbeTypeModifier(probe, s"Expected $probe to be a probe.")
    if (!probe.isWidthKnown) Builder.error("Probe width unknown.")
    val probeWidth = probe.widthOption.getOrElse(0)

    // check data width is known
    data.widthOption match {
      case None => Builder.error("Data width unknown.")
      case Some(w) =>
        if (probe.widthOption.exists(w > _)) Builder.error(s"Data width $w is larger than $probeWidth.")
    }

    data match {
      case d: Bits => d.pad(probeWidth).asInstanceOf[T]
      case d => d
    }
  }

  /** Override existing driver of a writable probe on initialization.
    *
    * @param probe writable Probe to force
    * @param value to force onto the probe
    */
  def forceInitial(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot forceInitial a non-writable Probe.")
    pushCommand(ProbeForceInitial(sourceInfo, probe.ref, padDataToProbeWidth(value, probe).ref))
  }

  /** Release initial driver on a probe.
    *
    * @param probe writable Probe to release
    */
  def releaseInitial(probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot releaseInitial a non-writable Probe.")
    pushCommand(ProbeReleaseInitial(sourceInfo, probe.ref))
  }

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
  def force(probe: Data, value: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot force a non-writable Probe.")
    val clock = Builder.forcedClock
    val cond = Module.disableOption.map(!_.value).getOrElse(true.B)
    pushCommand(ProbeForce(sourceInfo, clock.ref, cond.ref, probe.ref, padDataToProbeWidth(value, probe).ref))
  }

  /** Release driver on a probe. If called within the scope of a [[when]]
    * block, the release will only occur on cycles that the when condition
    * is true.
    *
    * Fires only when reset has been asserted and then deasserted through the
    * [[Disable]] API.
    *
    * @param probe writable Probe to release
    */
  def release(probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    requireHasWritableProbeTypeModifier(probe, "Cannot release a non-writable Probe.")
    val clock = Builder.forcedClock
    val cond = Module.disableOption.map(!_.value).getOrElse(true.B)
    pushCommand(ProbeRelease(sourceInfo, clock.ref, cond.ref, probe.ref))
  }

}
