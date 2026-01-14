// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Names
import dotty.tools.dotc.core.Constants.Constant
import dotty.tools.dotc.core.Types.*
import dotty.tools.dotc.core.Flags
import dotty.tools.dotc.plugins.PluginPhase
import dotty.tools.dotc.transform.PickleQuotes

import chisel3.internal.sourceinfo.SourceInfoFileResolver

/** Scala 3 compiler plugin phase that adds source locators to Module class definitions.
  *
  * For each non-abstract class that extends BaseModule, this phase adds an override of the
  * `_sourceInfo` method that returns a SourceLine with the file path, line number, and column
  * of the class definition.
  */
class ChiselSourceLocatorPhase extends PluginPhase {
  val phaseName: String = "chiselSourceLocatorPhase"
  override val runsAfter = Set(PickleQuotes.name)

  override def transformTypeDef(tree: tpd.TypeDef)(using Context): tpd.Tree = {
    if (
      tree.isClassDef
      && ChiselTypeHelpers.isModule(tree.tpe)
      && !tree.symbol.flags.is(Flags.Abstract)
      && !hasOverriddenSourceLocator(tree)
    ) {
      addSourceLocator(tree)
    } else {
      super.transformTypeDef(tree)
    }
  }

  /** Check if the class already has a _sourceInfo method override */
  private def hasOverriddenSourceLocator(tree: tpd.TypeDef)(using Context): Boolean = {
    val template = tree.rhs.asInstanceOf[tpd.Template]
    template.body.exists {
      case dd: tpd.DefDef => dd.name.toString == "_sourceInfo"
      case _ => false
    }
  }

  /** Add a _sourceInfo override to the class that returns the source location */
  private def addSourceLocator(tree: tpd.TypeDef)(using Context): tpd.Tree = {
    val sourceFile = tree.sourcePos.source.file
    val path = sourceFile.jpath match {
      case null  => sourceFile.path
      case jpath => SourceInfoFileResolver.resolve(jpath)
    }
    val line = tree.sourcePos.line + 1 // Convert from 0-indexed to 1-indexed
    val column = tree.sourcePos.column

    // Create the SourceLine expression: chisel3.experimental.SourceLine(path, line, column)
    val sourceLineClass = requiredClass("chisel3.experimental.SourceLine")
    val sourceInfoTpe = requiredClassRef("chisel3.experimental.SourceInfo")

    val sourceLineExpr = tpd.New(
      sourceLineClass.typeRef,
      List(
        tpd.Literal(Constant(path)),
        tpd.Literal(Constant(line)),
        tpd.Literal(Constant(column))
      )
    )

    // Create the _sourceInfo method: override protected def _sourceInfo: SourceInfo = ...
    val sourceInfoSym = newSymbol(
      tree.symbol,
      Names.termName("_sourceInfo"),
      Flags.Method | Flags.Override | Flags.Protected,
      MethodType(Nil, Nil, sourceInfoTpe)
    )

    val sourceInfoDef = tpd.DefDef(sourceInfoSym.asTerm, _ => sourceLineExpr)

    // Add the new method to the template body
    tree match {
      case td @ tpd.TypeDef(name, tmpl: tpd.Template) =>
        val newTemplate = tpd.cpy.Template(tmpl)(body = sourceInfoDef :: tmpl.body)
        tpd.cpy.TypeDef(td)(name, newTemplate)
      case _ => super.transformTypeDef(tree)
    }
  }
}
