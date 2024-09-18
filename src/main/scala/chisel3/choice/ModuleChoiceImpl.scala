// SPDX-License-Identifier: Apache-2.0

package chisel3.choice

import scala.collection.immutable.ListMap

import chisel3.{Data, FixedIOBaseModule, Module, SourceInfoDoc}
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.{groupByIntoSeq, Builder}
import chisel3.internal.binding.WireBinding
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir.DefInstanceChoice

private[chisel3] trait ModuleChoiceImpl {

  protected def _applyImpl[T <: Data](
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
