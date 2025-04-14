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
import dotty.tools.dotc.transform.{Erasure, Pickler, PostTyper}
import dotty.tools.dotc.core.Types.*
import dotty.tools.dotc.core.Flags
import dotty.tools.dotc.util.SourcePosition

import scala.collection.mutable

class ChiselComponent extends StandardPlugin {
  val name:                 String = "ChiselComponent"
  override val description: String = "Chisel's type-specific naming"

  override def init(options: List[String]): List[PluginPhase] = {
    (new ChiselComponentPhase) :: Nil
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

    val pluginModule = requiredModule("chisel3")
    val autoNameMethod = pluginModule.requiredMethod("withName")
    val prefixModule = requiredModule("chisel3.experimental.prefix")
    val prefixApplyMethod = prefixModule.requiredMethod("applyString")

    val sym = tree.symbol
    val tpt = tree.tpt.tpe
    val name = sym.name
    val rhs = tree.rhs

    val valName: String = tree.name.show
    val nameLiteral = Literal(Constant(valName))
    val prefixLiteral = if (valName.head == '_') Literal(Constant(valName.tail)) else Literal(Constant(valName))

    val isData = ChiselTypeHelpers.isData(tpt)
    val isNamedComp = isData || ChiselTypeHelpers.isNamed(tpt)
    val isPrefixed = isNamedComp || ChiselTypeHelpers.isPrefixed(tpt)

    if (!ChiselTypeHelpers.okVal(tree)) tree // Cannot name this, so skip
    else if (isData && ChiselTypeHelpers.inBundle(tree)) { // Data in a bundle
      val newRHS = transformFollowing(rhs)
      val named =
        tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tpt).appliedTo(nameLiteral).appliedTo(newRHS)
      cpy.ValDef(tree)(rhs = named)
    } else if (isData || isPrefixed) { // All other Data subtype instances
      val newRHS = transformFollowing(rhs)
      val prefixed =
        tpd.ref(prefixModule).select(prefixApplyMethod).appliedToType(tpt).appliedTo(prefixLiteral).appliedTo(newRHS)
      val named =
        if (isNamedComp) {
          tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tpt).appliedTo(nameLiteral).appliedTo(prefixed)
        } else prefixed
      cpy.ValDef(tree)(rhs = named)
    } else if (ChiselTypeHelpers.isModule(tpt) || ChiselTypeHelpers.isInstance(tpt)) { // Modules or instances
      val newRHS = transformFollowing(rhs)
      val named =
        tpd.ref(pluginModule).select(autoNameMethod).appliedToType(tpt).appliedTo(nameLiteral).appliedTo(newRHS)
      cpy.ValDef(tree)(rhs = named)
    } else {
      super.transformValDef(tree)
    }
  }
}
