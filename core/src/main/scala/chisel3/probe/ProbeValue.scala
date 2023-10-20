// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3.{Data, Output, SourceInfoDoc, Wire}
import chisel3.internal.{Builder, OpBinding}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.{ProbeDefine, ProbeExpr, RWProbeExpr}
import chisel3.experimental.{requireIsHardware, SourceInfo}
import chisel3.reflect.DataMirror.collectLeafMembers

import scala.language.experimental.macros
import chisel3.Aggregate

private[chisel3] sealed trait ProbeValueBase {
  protected def apply[T <: Data](source: T, writable: Boolean)(implicit sourceInfo: SourceInfo): T = {
    requireIsHardware(source)
    // construct probe to return with cloned info
    val clone = if (writable) RWProbe(source.cloneTypeFull) else Probe(source.cloneTypeFull)
    clone.bind(OpBinding(Builder.forcedUserModule, Builder.currentWhen))
    // create reference for clone
    val cloneRef = source match {
      case agg: Aggregate =>
        // intermediate probe to hook up to aggregate elements
        val intermediate = Wire(Output(if (writable) RWProbe(agg.cloneTypeFull) else Probe(agg.cloneTypeFull)))
        collectLeafMembers(intermediate).zip(collectLeafMembers(agg)).foreach {
          case (i, s) =>
            if (writable) {
              pushCommand(ProbeDefine(sourceInfo, i.ref, RWProbeExpr(s.ref)))
            } else {
              pushCommand(ProbeDefine(sourceInfo, i.ref, ProbeExpr(s.ref)))
            }
        }
        intermediate.suggestName("probe_value")
        intermediate.ref
      case s =>
        if (writable) {
          RWProbeExpr(s.ref)
        } else {
          ProbeExpr(s.ref)
        }
    }

    clone.setRef(cloneRef)
    clone
  }
}

object ProbeValue extends ProbeValueBase with SourceInfoDoc {

  /** Create a read-only probe expression. */
  def apply[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = super.apply(source, writable = false)
}

object RWProbeValue extends ProbeValueBase with SourceInfoDoc {

  /** Create a read/write probe expression. */
  def apply[T <: Data](source: T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: T)(implicit sourceInfo: SourceInfo): T = super.apply(source, writable = true)
}
