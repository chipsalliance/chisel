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

class ChiselNamingPhase extends PluginPhase {
  val phaseName: String = "chiselNamingPhase"
  override val runsAfter = Set(TyperPhase.name)

  private var dataTpe:    TypeRef = _
  private var memBaseTpe: TypeRef = _
  private var verifTpe:   TypeRef = _
  private var dynObjTpe:  TypeRef = _
  private var affectsTpe: TypeRef = _
  private var moduleTpe:  TypeRef = _
  private var instTpe:    TypeRef = _
  private var prefixTpe:  TypeRef = _
  private var bundleTpe:  TypeRef = _

  private var pluginModule:      TermSymbol = _
  private var autoNameMethod:    TermSymbol = _
  private var withNames:         TermSymbol = _
  private var prefixModule:      TermSymbol = _
  private var prefixApplyMethod: TermSymbol = _

  override def prepareForUnit(tree: Tree)(using ctx: Context): Context = {
    dataTpe = requiredClassRef("chisel3.Data")
    memBaseTpe = requiredClassRef("chisel3.MemBase")
    verifTpe = requiredClassRef("chisel3.VerificationStatement")
    dynObjTpe = requiredClassRef("chisel3.Disable")
    affectsTpe = requiredClassRef("chisel3.experimental.AffectsChiselName")
    moduleTpe = requiredClassRef("chisel3.experimental.BaseModule")
    instTpe = requiredClassRef("chisel3.experimental.hierarchy.Instance")
    prefixTpe = requiredClassRef("chisel3.experimental.AffectsChiselPrefix")
    bundleTpe = requiredClassRef("chisel3.Bundle")

    pluginModule = requiredModule("chisel3")
    autoNameMethod = pluginModule.requiredMethod("withName")
    withNames = pluginModule.requiredMethod("withNames")
    prefixModule = requiredModule("chisel3.experimental.prefix")
    prefixApplyMethod = prefixModule.requiredMethod("applyString")
    ctx
  }

  override def transformValDef(tree: tpd.ValDef)(using Context): tpd.Tree = {
    val sym = tree.symbol
    val tpt = tree.tpt.tpe
    val name = sym.name
    val rhs = tree.rhs

    val valName: String = tree.name.show
    val nameLiteral = Literal(Constant(valName))
    val prefixLiteral =
      if (valName.head == '_')
        Literal(Constant(valName.tail))
      else Literal(Constant(valName))

    val isData = ChiselTypeHelpers.isData(tpt)
    val isBoxedData = ChiselTypeHelpers.isBoxedData(tpt)
    val isNamedComp = isData || isBoxedData || ChiselTypeHelpers.isNamed(tpt)
    val isPrefixed = isNamedComp || ChiselTypeHelpers.isPrefixed(tpt)

    if (!ChiselTypeHelpers.okVal(tree)) tree // Cannot name this, so skip
    else if (isData && ChiselTypeHelpers.inBundle(sym)) { // Data in a bundle
      val newRHS = transformFollowing(rhs)
      val named =
        tpd
          .ref(pluginModule)
          .select(autoNameMethod)
          .appliedToType(tpt)
          .appliedTo(nameLiteral)
          .appliedTo(newRHS)
      cpy.ValDef(tree)(rhs = named)
    } else if (isData || isBoxedData || isPrefixed) { // All Data subtype instances
      val newRHS = transformFollowing(rhs)
      val prefixed =
        tpd
          .ref(prefixModule)
          .select(prefixApplyMethod)
          .appliedToType(tpt)
          .appliedTo(prefixLiteral)
          .appliedTo(newRHS)
      val named = if (isNamedComp) {
        tpd
          .ref(pluginModule)
          .select(autoNameMethod)
          .appliedToType(tpt)
          .appliedTo(nameLiteral)
          .appliedTo(prefixed)
      } else prefixed
      cpy.ValDef(tree)(rhs = named)
    } else if ( // Modules or instances
      ChiselTypeHelpers.isModule(tpt) ||
      ChiselTypeHelpers.isInstance(tpt)
    ) {
      val newRHS = transformFollowing(rhs)
      val named =
        tpd
          .ref(pluginModule)
          .select(autoNameMethod)
          .appliedToType(tpt)
          .appliedTo(nameLiteral)
          .appliedTo(newRHS)
      cpy.ValDef(tree)(rhs = named)
    } else {
      super.transformValDef(tree)
    }
  }
}
