// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.Data.ProbeInfo
import chisel3.experimental.SourceInfo
import chisel3.reflect.DataMirror.{checkTypeEquivalence, collectAllMembers, collectLeafMembers, hasProbeTypeModifier}

import scala.language.experimental.macros

package object probe extends SourceInfoDoc {

  /** Set the probe information of the [[Data]] — if dealing with an
    * [[Aggregate]], set the probe information for leaf elements only.
    */
  private[chisel3] def setProbeModifier[T <: Data](data: T, probeInfo: Option[ProbeInfo]): Unit = {
    probeInfo.foreach { _ =>
      collectAllMembers(data).foreach { e =>
        e match {
          // sample_element is used to determine the Vec element type in the converter
          case v: Vec[_]    => setProbeModifier(v.sample_element, probeInfo)
          case v: Aggregate => // do nothing
          case _ => e.probeInfo = probeInfo
        }
      }
    }
  }

  /** Initialize a probe with a provided probe value. */
  def define[T <: Data](sink: T, probeExpr: T)(implicit sourceInfo: SourceInfo): Unit = {
    if (!checkTypeEquivalence(sink, probeExpr)) {
      Builder.error("Cannot define a probe on a non-equivalent type.")
    }
    collectLeafMembers(sink).zip(collectLeafMembers(probeExpr)).foreach {
      case (s, pe) =>
        requireHasProbeTypeModifier(s, "Expected sink to be a probe.")
        requireHasProbeTypeModifier(pe, "Expected source to be a probe expression.")
        if (s.probeInfo.get.writable) {
          requireHasWritableProbeTypeModifier(
            pe,
            "Cannot use a non-writable probe expression to define a writable probe."
          )
        }
        pushCommand(ProbeDefine(sourceInfo, s.ref, pe.ref))
    }
  }

  /** Access the value of a probe. */
  def read[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceRead[T]

  /** @group SourceInfoTransformMacro */
  def do_read[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = {
    requireIsHardware(source)
    // construct clone to bind to ProbeRead
    val clone = source.cloneTypeFull
    clone.bind(OpBinding(Builder.forcedUserModule, Builder.currentWhen))
    val cloneRef = source match {
      case agg: Aggregate =>
        // intermediate wire to facilitate aggregate read
        val intermediate = Wire(Output(agg.cloneTypeFull))
        clearProbeInfo(intermediate)
        val sourceElems = collectLeafMembers(agg)
        val intermediateElems = collectLeafMembers(intermediate)
        sourceElems.zip(intermediateElems).foreach { case (e, t) => t :#= do_read(e) }
        intermediate.suggestName("probe_read")
        intermediate.ref
      case s =>
        requireHasProbeTypeModifier(s)
        ProbeRead(s.ref)
    }
    clone.setRef(cloneRef)
    // return a non-probe type Data that can be used in Data connects
    clearProbeInfo(clone)
    clone
  }

  /** Clear all ProbeInfo */
  private def clearProbeInfo[T <: Data](data: T): Unit = {
    collectLeafMembers(data).foreach { elem => elem.probeInfo = None }
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

  /** Override existing driver of a writable probe on initialization. */
  def forceInitial[T <: Data](probe: T, value: T)(implicit sourceInfo: SourceInfo): Unit = {
    collectLeafMembers(probe).zip(collectLeafMembers(value)).foreach {
      case (p, v) =>
        val padValue = padDataToProbeWidth(v, p)
        if (!checkTypeEquivalence(p, padValue)) {
          Builder.error("Cannot forceInitial a probe with a non-equivalent type.")
        }
        requireHasWritableProbeTypeModifier(p, "Cannot forceInitial a non-writable Probe.")
        pushCommand(ProbeForceInitial(sourceInfo, p.ref, padValue.ref))
    }
  }

  /** Release initial driver on a probe. */
  def releaseInitial(probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    collectLeafMembers(probe).foreach { p =>
      requireHasWritableProbeTypeModifier(p, "Cannot releaseInitial a non-writable Probe.")
      pushCommand(ProbeReleaseInitial(sourceInfo, p.ref))
    }
  }

  /** Override existing driver of a writable probe. */
  def force[T <: Data](clock: Clock, cond: Bool, probe: T, value: T)(implicit sourceInfo: SourceInfo): Unit = {
    collectLeafMembers(probe).zip(collectLeafMembers(value)).foreach {
      case (p, v) =>
        val padValue = padDataToProbeWidth(v, p)
        if (!checkTypeEquivalence(p, padValue)) {
          Builder.error("Cannot force a probe with a non-equivalent type.")
        }
        requireHasWritableProbeTypeModifier(p, "Cannot force a non-writable Probe.")
        pushCommand(ProbeForce(sourceInfo, clock.ref, cond.ref, p.ref, padValue.ref))
    }
  }

  /** Release driver on a probe. */
  def release(clock: Clock, cond: Bool, probe: Data)(implicit sourceInfo: SourceInfo): Unit = {
    collectLeafMembers(probe).foreach { p =>
      requireHasWritableProbeTypeModifier(p, "Cannot release a non-writable Probe.")
      pushCommand(ProbeRelease(sourceInfo, clock.ref, cond.ref, p.ref))
    }
  }

}
