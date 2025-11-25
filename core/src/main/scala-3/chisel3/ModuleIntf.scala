// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}

transparent private[chisel3] trait Module$Intf extends SourceInfoDoc { self: Module.type =>

  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  // TODO(adkian-sifive) the callsite here explicitly passes
  // sourceInfo so it cannot be a contextual parameter
  transparent inline def apply[T <: BaseModule](inline bc: => T): T = _applyImpl(bc)
}
