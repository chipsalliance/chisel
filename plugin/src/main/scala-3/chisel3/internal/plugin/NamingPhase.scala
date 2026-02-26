// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import dotty.tools.dotc.*
import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.ast.tpd.*
import dotty.tools.dotc.ast.Trees
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Names.{termName, Name, TermName}
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

  // Map of synthetic unapply temp names like $1$ to list of extracted names
  private val unapplyNamesMap = mutable.Map[Name, List[TermName]]()

  // Empty name constant for missing tuple elements
  private val EmptyName = termName("")

  // Scan a list of statements to find unapply patterns and collect name mappings.
  // In Scala 3, simple tuple unapply like `val (a, b) = expr` is desugared to:
  //   val $1$ = expr
  //   val a = $1$._1
  //   val b = $1$._2
  private def collectUnapplyNames(stats: List[tpd.Tree])(using Context): Unit = {
    // Pattern to match tuple element accessors like _1, _2, etc.
    val TupleIndex = "_([0-9]+)".r

    val syntheticTupleVals = mutable.Map[Name, Int]()
    val selectsByQualifier = mutable.Map[Name, mutable.ListBuffer[(Int, TermName)]]()

    stats.foreach {
      case vd: tpd.ValDef =>
        if (ChiselTypeHelpers.okUnapply(vd) && !vd.rhs.isInstanceOf[Match]) {
          // Synthetic tuple val from unapply
          syntheticTupleVals(vd.name) = ChiselTypeHelpers.tupleArity(vd.tpt.tpe)
        } else {
          // Check for tuple element select like: val a = $1$._1
          vd.rhs match {
            // Extract qualifier name, handling both local (Ident) and member (Select) access
            case Select(qual, selectedName) if ChiselTypeHelpers.isTupleType(qual.tpe) =>
              val qualName: Option[Name] = qual match {
                case Ident(n)           => Some(n)
                case Select(This(_), n) => Some(n)
                case Select(_, n)       => Some(n)
                case _                  => None
              }
              qualName.foreach { qn =>
                selectedName.toString match {
                  case TupleIndex(idx) =>
                    selectsByQualifier.getOrElseUpdate(qn, mutable.ListBuffer()) += ((idx.toInt - 1, vd.name))
                  case _ => ()
                }
              }
            case _ => ()
          }
        }
      case _ => ()
    }

    // Build unapplyNamesMap from collected data
    syntheticTupleVals.foreach { case (syntheticName, arity) =>
      selectsByQualifier.get(syntheticName).foreach { selects =>
        val indexedNames = selects.filter { case (idx, _) => idx >= 0 && idx < arity }
        if (indexedNames.nonEmpty) {
          val names = (0 until arity).map(i => indexedNames.find(_._1 == i).map(_._2).getOrElse(EmptyName)).toList
          unapplyNamesMap(syntheticName) = names
        }
      }
    }
  }

  override def prepareForUnit(tree: Tree)(using ctx: Context): Context = {
    unapplyNamesMap.clear() // Clear for new compilation unit
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

  override def prepareForBlock(tree: tpd.Block)(using Context): Context = {
    collectUnapplyNames(tree.stats :+ tree.expr)
    ctx
  }

  override def prepareForTemplate(tree: tpd.Template)(using Context): Context = {
    collectUnapplyNames(tree.body)
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

    // Check if this is an unapply pattern (tuple destructuring)
    val isSyntheticUnapply = ChiselTypeHelpers.okUnapply(tree)
    val isMatchUnapply = isSyntheticUnapply && rhs.isInstanceOf[Match]
    val isSimpleUnapply = isSyntheticUnapply && !rhs.isInstanceOf[Match]

    if (isMatchUnapply) {
      // Handle case class unapply like: val Foo(a, b) = func()
      // Names are extracted from the Match pattern
      val fieldsOfInterest = ChiselTypeHelpers.tupleFieldsOfInterest(tpt)
      if (fieldsOfInterest.nonEmpty && fieldsOfInterest.exists(identity)) {
        ChiselTypeHelpers.findUnapplyNames(rhs) match {
          case Some(names) =>
            // Only name fields that are NamedComponents
            val onames = fieldsOfInterest.zip(names).map { case (ok, n) => if (ok) n else "" }
            val newRHS = transformFollowing(rhs)
            // Create string literals for each name
            val nameLiterals = onames.map(n => Literal(Constant(n)))
            val named =
              tpd
                .ref(pluginModule)
                .select(withNames)
                .appliedToType(tpt)
                .appliedToVarargs(nameLiterals, tpd.TypeTree(defn.StringType))
                .appliedTo(newRHS)
            cpy.ValDef(tree)(rhs = named)
          case None =>
            super.transformValDef(tree)
        }
      } else {
        super.transformValDef(tree)
      }
    } else if (isSimpleUnapply) {
      // Handle simple tuple unapply like: val (a, b) = (Wire(...), Wire(...))
      // Names are collected from subsequent selecting ValDefs via prepareForBlock
      val fieldsOfInterest = ChiselTypeHelpers.tupleFieldsOfInterest(tpt)
      unapplyNamesMap.get(tree.name) match {
        case Some(names) if fieldsOfInterest.nonEmpty && fieldsOfInterest.exists(identity) =>
          // Only name fields that are NamedComponents
          val onames = fieldsOfInterest.zip(names).map { case (ok, n) => if (ok) n else EmptyName }
          val newRHS = transformFollowing(rhs)
          val nameLiterals = onames.map(n => Literal(Constant(n.toString)))
          val named =
            tpd
              .ref(pluginModule)
              .select(withNames)
              .appliedToType(tpt)
              .appliedToVarargs(nameLiterals, tpd.TypeTree(defn.StringType))
              .appliedTo(newRHS)
          cpy.ValDef(tree)(rhs = named)
        case _ =>
          super.transformValDef(tree)
      }
    } else if (!ChiselTypeHelpers.okVal(tree)) tree // Cannot name this, so skip
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
