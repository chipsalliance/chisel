// SPDX-License-Identifier: Apache-2.0

package chisel3.choice

import scala.collection.immutable.ListMap

import chisel3.{Data, FixedIOBaseModule, Module, SourceInfoDoc}
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.experimental.hierarchy.core.Lookupable
import chisel3.internal.{groupByIntoSeq, Builder}
import chisel3.internal.binding.InstanceChoiceBinding
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir.DefInstanceChoice

object ModuleChoice extends ModuleChoiceObjIntf {

  protected def _applyImpl[T](
    default: => FixedIOBaseModule[T],
    choices: Seq[(Case, () => FixedIOBaseModule[T])]
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    val instDefaultModule = Module.evaluate(default)
    val lk: Lookupable[T] = instDefaultModule._lookupable

    val defaultLeaves = lk.in(instDefaultModule.io)

    val choiceModules = choices.map { case (choice, module) =>
      val instModule = Module.evaluate(module())
      val choiceLeaves = lk.in(instModule.io)
      if (!defaultLeaves.zip(choiceLeaves).forall { case (a, b) => b.typeEquivalent(a) }) {
        Builder.error("Error: choice module IO bundles are not type equivalent")
      }
      Builder.options += choice
      (choice, instModule)
    }

    groupByIntoSeq(choiceModules.map(_._1))(opt => opt).foreach { case (_, group) =>
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

    val bindingLeaves = defaultLeaves.map(_.cloneTypeFull)
    bindingLeaves.foreach(_.bind(InstanceChoiceBinding(Builder.forcedUserModule, Builder.currentBlock)))
    val binding = lk.out(instDefaultModule.io, bindingLeaves.iterator)

    pushCommand(
      DefInstanceChoice(
        sourceInfo,
        bindingLeaves.head,
        instDefaultModule,
        group,
        choiceModules.map { case (choice, instModule) =>
          (choice.name, instModule)
        }
      )
    )

    binding
  }
}
