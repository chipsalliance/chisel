// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.sourceinfo.InstTransform

object Module extends ObjectModuleImpl with SourceInfoDoc {

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
}

/** Abstract base class for Modules, which behave much like Verilog modules.
  * These may contain both logic and state which are written in the Module
  * body (constructor).
  * This abstract base class includes an implicit clock and reset.
  *
  * @note Module instantiations must be wrapped in a Module() call.
  */
abstract class Module extends ModuleImpl
