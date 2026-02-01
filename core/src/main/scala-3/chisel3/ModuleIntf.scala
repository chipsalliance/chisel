// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}

private[chisel3] trait Module$Intf extends SourceInfoDoc { self: Module.type =>

  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule](bc: => T)(using SourceInfo): T = _applyImpl(bc)

  /** Returns the implicit Disable
    *
    * Note that [[Disable]] is a function of the implicit clock and reset
    * so having no implicit clock or reset may imply no `Disable`.
    */
  def disable(using SourceInfo): Disable = _disableImpl

  /** Returns the current implicit [[Disable]], if one is defined
    *
    * Note that [[Disable]] is a function of the implicit clock and reset
    * so having no implicit clock or reset may imply no `Disable`.
    */
  def disableOption(using SourceInfo): Option[Disable] = _disableOptionImpl
}
