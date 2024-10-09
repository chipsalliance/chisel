// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3.{Data, SourceInfoDoc}
import chisel3.experimental.SourceInfo

import scala.language.experimental.macros

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
