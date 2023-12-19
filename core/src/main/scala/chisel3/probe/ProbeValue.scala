// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3.{Data, SourceInfoDoc}
import chisel3.internal.{Builder, OpBinding}
import chisel3.internal.firrtl.{ProbeExpr, RWProbeExpr}
import chisel3.experimental.{requireIsHardware, SourceInfo}

import scala.language.experimental.macros

object ProbeValue extends SourceInfoDoc {

  /** Create a read-only probe expression connected to an existing hardware element
   *
   * @param source the hardware element you want to probe
   * @return the Probe connectd to source
  */
  private[chisel3] def apply[T <: Data](source: T): Probe[T] = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_apply[T <: Data](source: T)(implicit sourceInfo: SourceInfo): Probe[T] = {
    requireIsHardware(source)
    // construct probe to return with cloned info
    val clone = Probe(source.cloneType)
    clone.underlying.bind(OpBinding(Builder.forcedUserModule, Builder.currentWhen))
    clone.underlying.setRef(ProbeExpr(source.ref))
    clone
  }
}

object RWProbeValue extends SourceInfoDoc {

  /** Create a read/write probe expression connected to an existing hardware element
   * 
   * @param source the hardware element you want to probe
   * @return the RWProbe connectd to source
  */
  private [chisel3] def apply[T <: Data](source: T): RWProbe[T] = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  private [chisel3] def do_apply[T <: Data](source: T)(implicit sourceInfo: SourceInfo): RWProbe[T] = {
    requireIsHardware(source)
    // construct probe to return with cloned info
    val clone = RWProbe(source.cloneType)
    clone.underlying.bind(OpBinding(Builder.forcedUserModule, Builder.currentWhen))
    clone.underlying.setRef(RWProbeExpr(source.ref))
    clone
  }
}
