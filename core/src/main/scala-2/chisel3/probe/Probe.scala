// SPDX-License-Identifier: Apache-2.0

package chisel3.probe

import chisel3._
import chisel3.experimental.SourceInfo

import scala.language.experimental.macros

object Probe extends ProbeBase with SourceInfoDoc {

  /** Mark a Chisel type as with a probe modifier.
    */
  def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  def apply[T <: Data](source: => T, color: layer.Layer): T =
    macro chisel3.internal.sourceinfo.ProbeTransform.sourceApplyWithColor[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T =
    super.apply(source, false, None)

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: => T, color: Option[layer.Layer])(implicit sourceInfo: SourceInfo): T =
    super.apply(source, false, color)
}

object RWProbe extends ProbeBase with SourceInfoDoc {

  /** Mark a Chisel type with a writable probe modifier.
    */
  def apply[T <: Data](source: => T): T = macro chisel3.internal.sourceinfo.ProbeTransform.sourceApply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = super.apply(source, true)
}
