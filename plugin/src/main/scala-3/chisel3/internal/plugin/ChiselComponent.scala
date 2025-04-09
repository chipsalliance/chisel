// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import dotty.tools.dotc.*
import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.ast.tpd.*
import dotty.tools.dotc.ast.Trees
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Names.TermName
import dotty.tools.dotc.core.StdNames.*
import dotty.tools.dotc.core.Constants.Constant
import dotty.tools.dotc.typer.TyperPhase
import dotty.tools.dotc.plugins.{PluginPhase, StandardPlugin}
import dotty.tools.dotc.transform.{Pickler, PostTyper, Erasure}
import dotty.tools.dotc.core.Types.*
import dotty.tools.dotc.core.Flags
import dotty.tools.dotc.util.SourcePosition

import scala.annotation.tailrec
import scala.collection.mutable

class ChiselComponent extends StandardPlugin {
  val name: String = "ChiselComponent"
  override val description: String = "Chisel's type-specific naming"

  override def init(options: List[String]): List[PluginPhase] = {
    (new ChiselComponentPhase) :: Nil
  }
}

@tailrec
def printTreeString(t: String, accum: String = "", indent: Int = 0): String = {
  val parenop: Option[Char] = t.find { c => c == '(' || c == ')'}
  parenop match {
    case Some('(') => {
      val splitstr = t.span { c => c != parenop.get }
      printTreeString(
        splitstr._2 stripPrefix("("),
        accum ++ (" " * indent) ++ (splitstr._1 stripPrefix(",")) ++ "\n",
        indent + 2
      )
    }
    case Some(')') => {
      val splitstr = t.span { c => c != parenop.get }
      def newstr: String = splitstr._1 match {
        case "" => accum
        case _ => accum ++ (" " * indent) ++ (splitstr._1 stripPrefix(",")) ++ "\n"
      }
      printTreeString(
        splitstr._2 stripPrefix(")"),
        newstr,
        indent - 2
      )
    }
    case _ => accum
  }
}

class ChiselComponentPhase extends PluginPhase {

  val phaseName: String = "chiselComponentPhase"
  override val runsAfter = Set(TyperPhase.name)

  override def transformValDef(tree: tpd.ValDef)(using Context): tpd.Tree = {
    val dataTpe = requiredClassRef("chisel3.Data")
    val memBaseTpe = requiredClassRef("chisel3.MemBase")
    val verifTpe = requiredClassRef("chisel3.VerificationStatement")
    val dynObjTpe = requiredClassRef("chisel3.Disable")
    val affectsTpe = requiredClassRef("chisel3.experimental.AffectsChiselName")
    val moduleTpe = requiredClassRef("chisel3.experimental.BaseModule")
    val instTpe = requiredClassRef("chisel3.experimental.hierarchy.Instance")
    val prefixTpe = requiredClassRef("chisel3.experimental.AffectsChiselPrefix")
    val bundleTpe = requiredClassRef("chisel3.Bundle")

    val pluginModule = requiredModule("chisel3.internal.plugin")
    val autoNameMethod = pluginModule.requiredMethod("autoNameRecursively")
    val autoNameProductMethod = pluginModule.requiredMethod("autoNameRecursivelyProduct")
    val prefixModule = requiredModule("chisel3.experimental.prefix")
    val prefixApplyMethod = prefixModule.requiredMethod("applyString")
    
    val sym = tree.symbol
    val tpt = tree.tpt.tpe
    val name = sym.name
    val rhs = tree.rhs

    val valName: String = tree.name.show
    val nameLiteral = Literal(Constant(valName))
    val prefixLiteral = if (valName.head == '_') Literal(Constant(valName.tail)) else Literal(Constant(valName))

    val isData = ChiselTypeHelpers.shouldMatchData(tpt)
    val isNamedComp = isData || ChiselTypeHelpers.shouldMatchNamedComp(tpt)
    val isPrefixed = isNamedComp || ChiselTypeHelpers.shouldMatchChiselPrefixed(tpt)

    if (isData && ChiselTypeHelpers.inBundle(tree)) {
      val newRHS = transformFollowing(rhs)
      val named = tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tpt).appliedTo(nameLiteral).appliedTo(newRHS)
      cpy.ValDef(tree)(rhs = named)

    } else if (isData || isPrefixed) {
      val newRHS = transformFollowing(rhs)
      val prefixed = tpd.ref(prefixModule).select(prefixApplyMethod).appliedToType(tpt).appliedTo(prefixLiteral).appliedTo(newRHS)
      val named =
        if (isNamedComp) {
          tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tpt).appliedTo(nameLiteral).appliedTo(prefixed)
        } else prefixed
      cpy.ValDef(tree)(rhs = named)

    } else if (ChiselTypeHelpers.shouldMatchModule(tpt) || ChiselTypeHelpers.shouldMatchInstance(tpt)) {
      val newRHS = transformFollowing(rhs)
      val named = tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tpt).appliedTo(nameLiteral).appliedTo(newRHS)
      cpy.ValDef(tree)(rhs = named)

    } else {
      super.transformValDef(tree)
    }
  }

  // override def transformClassDef(tree: tpd.TypeDef)(using Context): tpd.Tree = {
  //   val dataTpe = requiredClassRef("chisel3.Data")
  //   val memBaseTpe = requiredClassRef("chisel3.MemBase")
  //   val verifTpe = requiredClassRef("chisel3.VerificationStatement")
  //   val dynObjTpe = requiredClassRef("chisel3.Disable")
  //   val affectsTpe = requiredClassRef("chisel3.experimental.AffectsChiselName")
  //   val moduleTpe = requiredClassRef("chisel3.experimental.BaseModule")
  //   val instTpe = requiredClassRef("chisel3.experimental.hierarchy.Instance")
  //   val prefixTpe = requiredClassRef("chisel3.experimental.AffectsChiselPrefix")
  //   val bundleTpe = requiredClassRef("chisel3.Bundle")

  //   val pluginModule = requiredModule("chisel3.internal.plugin")
  //   val autoNameMethod = pluginModule.requiredMethod("autoNameRecursively")
  //   val autoNameProductMethod = pluginModule.requiredMethod("autoNameRecursivelyProduct")
  //   val prefixModule = requiredModule("chisel3.experimental.prefix")
  //   val prefixApplyMethod = prefixModule.requiredMethod("applyString")
    
  //   val sym = tree.symbol
  //   if (isAModule(sym) && !sym.flags.is(Flags.Abstract) && !isOverriddenSourceLocator(tree.impl)) {
  //     val pos = tree.sourcePos
  //     val path = SourceInfoFileResolver.resolve(pos.source)
  //     val infoTree = Apply(
  //       Select(Select(Ident("chisel3"), "experimental"), "SourceLine"),
  //       List(Literal(Constant(path)), Literal(Constant(pos.line)), Literal(Constant(pos.column)))
  //     )
  //     val sourceInfoSym = newSymbol(sym, TermName("_sourceInfo"), Flags.Override | Flags.Protected, MethodType(Nil, sourceInfoTpe))
  //     val sourceInfoDef = DefDef(sourceInfoSym, infoTree)

  //     val newTemplate = cpy.Template(tree.impl)(
  //       body = sourceInfoDef :: tree.impl.body
  //     )
  //     cpy.ClassDef(tree)(impl = newTemplate)
  //   } else {
  //     super.transformClassDef(tree)
  //   }
  // }
}
