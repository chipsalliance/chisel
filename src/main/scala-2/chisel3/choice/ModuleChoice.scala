// SPDX-License-Identifier: Apache-2.0

package chisel3.choice

import scala.language.experimental.macros

import chisel3.{Data, FixedIOBaseModule, SourceInfoDoc}
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.InstChoiceTransform

object ModuleChoice extends ModuleChoiceImpl with SourceInfoDoc {

  /** A wrapper method for Module instantiation based on option choices.
    * (necessary to help Chisel track internal state).
    * @see [[chisel3.choice.Group]] on how to declare the options for the alternatives.
    *
    * Example:
    * ```
    * val module = ModuleChoice(new DefaultModule)(Seq(
    *   Platform.FPGA -> new FPGATarget,
    *   Platform.ASIC -> new ASICTarget
    * ))
    * ```
    *
    * @param default the Module to instantiate if no module is specified
    * @param choices the mapping from cases to module generators
    *
    * @return the input module `m` with Chisel metadata properly set
    *
    * @throws java.lang.IllegalArgumentException if the cases do not belong to the same option.
    */
  def apply[T <: Data](
    default: => FixedIOBaseModule[T]
  )(choices: => Seq[(Case, () => FixedIOBaseModule[T])]
  ): T =
    macro InstChoiceTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    default: => FixedIOBaseModule[T],
    choices: Seq[(Case, () => FixedIOBaseModule[T])]
  )(
    implicit sourceInfo: SourceInfo
  ): T = _applyImpl(default, choices)
}
