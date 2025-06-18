// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import dotty.tools.dotc.*
import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.ast.untpd
import dotty.tools.dotc.ast.tpd.*
import dotty.tools.dotc.ast.Trees
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Names
import dotty.tools.dotc.core.StdNames.*
import dotty.tools.dotc.core.Constants.Constant
import dotty.tools.dotc.typer.TyperPhase
import dotty.tools.dotc.plugins.{PluginPhase, StandardPlugin}
import dotty.tools.dotc.transform.{Erasure, Pickler, PostTyper}
import dotty.tools.dotc.core.Types
import dotty.tools.dotc.core.Types.*
import dotty.tools.dotc.core.Flags
import dotty.tools.dotc.util.SourcePosition
import dotty.tools.dotc.core.Decorators.toTermName
import dotty.tools.dotc.core.Definitions

import scala.collection.mutable

object BundleHelpers {
  def cloneTypeFull(tree: Tree)(using Context): Tree = {
    val cloneSym =
      requiredMethod("chisel3.reflect.DataMirror.internal.chiselTypeClone")
    tpd.Apply(
      tpd.TypeApply(
        tpd.ref(cloneSym),
        List(tpd.TypeTree(tree.tpe))
      ),
      List(tree)
    )
  }

  def generateAutoCloneType(
    record:     tpd.TypeDef,
    thiz:       tpd.This,
    conArgsOpt: Option[List[List[tpd.Tree]]],
    isBundle:   Boolean
  )(using Context): Option[tpd.DefDef] = {
    conArgsOpt.flatMap { conArgs =>
      val newExpr = tpd.New(record.symbol.typeRef, conArgs.flatten)
      val recordTpe = requiredClassRef("chisel3.Record")
      val cloneTypeSym = newSymbol(
        record.symbol,
        Names.termName("_cloneTypeImpl"),
        Flags.Method | Flags.Override | Flags.Protected,
        MethodType(Nil, Nil, recordTpe)
      )
      Some(tpd.DefDef(cloneTypeSym.asTerm, _ => newExpr))
    }
  }

  def extractConArgs(
    record:   tpd.TypeDef,
    thiz:     tpd.This,
    isBundle: Boolean
  )(using Context): Option[List[List[tpd.Tree]]] = {
    val template = record.rhs.asInstanceOf[tpd.Template]
    val primaryConstructorOpt = Option(template.constr) // ← pick the stored constructor

    val paramAccessors = record.symbol.primaryConstructor.paramSymss.flatten

    if (primaryConstructorOpt.isEmpty) {
      report.warning("Unable to determine primary constructor!", record.sourcePos)
      return None
    }

    val constructor = primaryConstructorOpt.get
    val paramLookup = paramAccessors.map(sym => sym.name.toString -> sym).toMap

    Some(constructor.termParamss.map(_.map { vp =>
      val p = paramLookup(vp.name.toString)
      val select = tpd.Select(thiz, p.name)
      val cloned: tpd.Tree =
        if (ChiselTypeHelpers.isData(vp.tpt.tpe)) cloneTypeFull(select) else select

      if (vp.tpt.tpe.isRepeatedParam)
        tpd.SeqLiteral(List(cloned), cloned)
      else cloned
    }))
  }

  private def makeArray(values: List[Tree])(using Context): Tree = {
    val elemTpe = defn.TupleClass.typeRef.appliedTo(List(defn.StringType, defn.AnyType))
    val vectorModule = ref(requiredModule("scala.collection.immutable.Vector").termRef)

    val vec = Apply(
      TypeApply(Select(vectorModule, nme.apply), List(TypeTree(elemTpe))),
      List(SeqLiteral(values, TypeTree(elemTpe)))
    )
    vec
  }

  /** Creates the tuple containing the given elements */
  def tupleTree(elems: List[Tree])(using Context): Tree = {
    val arity = elems.length
    if arity == 0 then ref(defn.EmptyTupleModule)
    else if arity <= Definitions.MaxTupleArity then
      // TupleN[elem1Tpe, ...](elem1, ...)
      ref(defn.TupleType(arity).nn.typeSymbol.companionModule)
        .select(nme.apply)
        .appliedToTypes(elems.map(_.tpe.widenIfUnstable))
        .appliedToArgs(elems)
    else
      // TupleXXL.apply(elems*) // TODO add and use Tuple.apply(elems*) ?
      ref(defn.TupleXXLModule)
        .select(nme.apply)
        .appliedToVarargs(elems.map(_.asInstance(defn.ObjectType)), TypeTree(defn.ObjectType))
        .asInstance(defn.tupleType(elems.map(elem => elem.tpe.widenIfUnstable)))
  }

  def getBundleFields(record: tpd.TypeDef)(using Context): List[Tree] = {
    val bundleSym = record.symbol.asClass
    val recordTpe = requiredClass("chisel3.Record")
    def isBundleDataField(m: Symbol): Boolean = {
      m.isPublic
      && (
        ChiselTypeHelpers.isData(m.info)
          || ChiselTypeHelpers.isBoxedData(m.info)
      )
    }

    val currentFields: List[Tree] = bundleSym.info.decls.toList.collect {
      case m if isBundleDataField(m) =>
        val name = m.name.show.trim
        val thisRef: tpd.Tree = tpd.This(bundleSym.asClass)
        val sel:     tpd.Tree = tpd.Select(thisRef, m.termRef)
        tupleTree(List(tpd.Literal(Constant(name)), sel))
    }
    currentFields
  }

  def generateElements(record: tpd.TypeDef)(using Context): tpd.DefDef = {
    val bundleSym = record.symbol.asClass

    val currentFields: List[Tree] = getBundleFields(record)

    val dataTpe = requiredClassRef("chisel3.Data")
    val rhs = makeArray(currentFields)

    val tupleTpe =
      defn.TupleClass.typeRef.appliedTo(List(defn.StringType, defn.AnyType))

    val iterableTpe =
      requiredClassRef("scala.collection.Iterable").appliedTo(tupleTpe)

    val elementsSym: Symbol =
      newSymbol(
        bundleSym,
        Names.termName("_elementsImpl"),
        Flags.Method | Flags.Override | Flags.Protected,
        MethodType(Nil, Nil, iterableTpe)
      )

    tpd.DefDef(elementsSym.asTerm, rhs)
  }
}

class ChiselBundlePhase extends PluginPhase {
  val phaseName: String = "chiselBundlePhase"
  override val runsAfter = Set(TyperPhase.name)

  override def transformTypeDef(record: tpd.TypeDef)(using Context): tpd.Tree = {
    if (
      ChiselTypeHelpers.isRecord(record.tpe)
      && !record.symbol.flags.is(Flags.Abstract)
    ) {
      val isBundle: Boolean = ChiselTypeHelpers.isBundle(record.tpe)
      val thiz:     tpd.This = tpd.This(record.symbol.asClass)

      // // ==================== Generate _cloneTypeImpl ====================
      val conArgs: Option[List[List[tpd.Tree]]] =
        BundleHelpers.extractConArgs(record, thiz, isBundle)
      val cloneTypeImplOpt =
        BundleHelpers.generateAutoCloneType(record, thiz, conArgs, isBundle)

      // ==================== Generate val elements (Bundles only) ====================
      val elementsImplOpt: Option[tpd.DefDef] =
        if (isBundle) Some(BundleHelpers.generateElements(record)) else None

      // ==================== Generate _usingPlugin ====================
      val usingPluginOpt =
        if (isBundle) {
          val isPluginSym = newSymbol(
            record.symbol,
            Names.termName("_usingPlugin"),
            Flags.Method | Flags.Override | Flags.Protected,
            defn.BooleanType
          )
          Some(
            tpd.DefDef(
              isPluginSym.asTerm,
              _ => tpd.Literal(Constant(true))
            )
          )
        } else None

      // TODO
      // val autoTypenameOpt =
      //   if (BundleHelpers.isAutoTypenamed(record.symbol)) {
      //     BundleHelpers.generateAutoTypename(record, thiz, conArgs.map(_.flatten))
      //   } else None

      record match {
        case td @ TypeDef(name, tmpl: tpd.Template) => {
          val newDefs =
            elementsImplOpt.toList ++ usingPluginOpt.toList ++ cloneTypeImplOpt.toList
          val newTemplate =
            if (tmpl.body.size >= 1)
              cpy.Template(tmpl)(body = newDefs ++ tmpl.body)
            else
              cpy.Template(tmpl)(body = newDefs)
          tpd.cpy.TypeDef(td)(name, newTemplate)
        }
        case _ => super.transformTypeDef(record)
      }
    } else {
      super.transformTypeDef(record)
    }
  }
}
