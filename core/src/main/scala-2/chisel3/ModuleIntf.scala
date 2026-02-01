// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.sourceinfo.InstTransform

private[chisel3] trait Module$Intf extends SourceInfoDoc { self: Module.type =>

  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule](bc: => T): T = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule](bc: => T)(implicit sourceInfo: SourceInfo): T = _applyImpl(bc)

  /** Returns the implicit Disable
    *
    * Note that [[Disable]] is a function of the implicit clock and reset
    * so having no implicit clock or reset may imply no `Disable`.
    */
  def disable(implicit sourceInfo: SourceInfo): Disable = _disableImpl

  /** Returns the current implicit [[Disable]], if one is defined
    *
    * Note that [[Disable]] is a function of the implicit clock and reset
    * so having no implicit clock or reset may imply no `Disable`.
    */
  def disableOption(implicit sourceInfo: SourceInfo): Option[Disable] = _disableOptionImpl
}
