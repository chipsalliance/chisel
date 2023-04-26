// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.Data.ProbeInfo
import chisel3.internal.{containsProbe, requireIsChiselType, requireNoProbeTypeModifier, Builder}
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
    if (containsProbe(data)) {
      Builder.error("Cannot create a probe of an aggregate containing a probe.")
    }
    if (writable && data.isConst) {
      Builder.error("Cannot create a writable probe of a const type.")
    }

    val ret: T = if (!data.mustClone(prevId)) data else data.cloneType.asInstanceOf[T]
    setProbeModifier(ret, Some(ProbeInfo(writable)))
    ret
  }
}

object Probe extends ProbeBase {

  /** Mark a Chisel type as with a probe modifier.
    */
  def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = super.apply(source, false)
}

object RWProbe extends ProbeBase {

  /** Mark a Chisel type with a writable probe modifier.
    */
  def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = super.apply(source, true)
}
