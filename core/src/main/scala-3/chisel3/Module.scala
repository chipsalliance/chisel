// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}

object Module extends ObjectModuleImpl with SourceInfoDoc {

  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  // TODO(adkian-sifive) the callsite here explicitly passes
  // sourceInfo so it cannot be a contextual parameter
  def apply[T <: BaseModule](bc: => T): T = _applyImpl(bc)
}

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  * This abstract base class includes an implicit clock and reset.
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class Module extends ModuleImpl
