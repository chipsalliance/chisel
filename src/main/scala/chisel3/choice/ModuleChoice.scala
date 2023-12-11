// SPDX-License-Identifier: Apache-2.0

package chisel3.choice

import scala.language.experimental.macros
import scala.collection.immutable.ListMap

import chisel3.{Data, FixedIOBaseModule, Module, SourceInfoDoc}
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.{groupByIntoSeq, Builder, WireBinding}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.DefInstanceChoice
import chisel3.internal.sourceinfo.InstChoiceTransform

object ModuleChoice extends SourceInfoDoc {

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
  ): T = {
    val instDefaultModule = Module.evaluate(default)

    val choiceModules = choices.map {
      case (choice, module) =>
        val instModule = Module.evaluate(module())
        if (!instModule.io.typeEquivalent(instDefaultModule.io)) {
          Builder.error("Error: choice module IO bundles are not type equivalent")
        }
        Builder.options += choice
        (choice, instModule)
    }

    groupByIntoSeq(choiceModules.map(_._1))(opt => opt).foreach {
      case (_, group) =>
        if (group.size != 1) {
          throw new IllegalArgumentException(s"Error: duplicate case '${group.head.name}'")
        }
    }

    val groupedChoices = choices.map(_._1.group).distinct
    if (groupedChoices.size == 0) {
      throw new IllegalArgumentException("Error: at least one alternative must be specified")
    }
    if (groupedChoices.size != 1) {
      val groupNames = groupedChoices.map(_.name).mkString(", ")
      throw new IllegalArgumentException(s"Error: cannot mix choices from different groups: ${groupNames}")
    }
    val group = groupedChoices.head.name

    val binding = instDefaultModule.io.cloneTypeFull
    binding.bind(WireBinding(Builder.forcedUserModule, Builder.currentWhen))

    pushCommand(
      DefInstanceChoice(
        sourceInfo,
        binding,
        instDefaultModule,
        group,
        choiceModules.map {
          case (choice, instModule) =>
            (choice.name, instModule)
        }
      )
    )

    binding
  }
}
