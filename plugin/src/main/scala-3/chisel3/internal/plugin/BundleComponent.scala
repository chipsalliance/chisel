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
    record: tpd.TypeDef,
    thiz: tpd.This,
    conArgsOpt: Option[List[List[tpd.Tree]]],
    isBundle: Boolean
  )(using Context): Option[tpd.DefDef] = {
    conArgsOpt.flatMap { conArgs =>
      val newExpr = tpd.New(record.symbol.typeRef, conArgs.flatten)

      val cloneTypeSym = newSymbol(
        record.symbol.owner,
        Names.termName("_cloneTypeImpl"),
        Flags.Method | Flags.Override | Flags.Protected,
        MethodType(Nil)(_ => Nil, _ => record.symbol.typeRef)
      )

      Some(tpd.DefDef(cloneTypeSym.asTerm, _ => newExpr))
    }
  }

  def extractConArgs(
    record: tpd.TypeDef,
    thiz: tpd.This,
    isBundle: Boolean
  )(using Context): Option[List[List[tpd.Tree]]] = {
    val body = record.rhs.asInstanceOf[tpd.Template].body

    val primaryConstructorOpt = body.collectFirst {
      case dd: tpd.DefDef if dd.symbol.isPrimaryConstructor => dd
    }

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
      val cloned: tpd.Tree = if (ChiselTypeHelpers.isData(vp.tpt.tpe))
        cloneTypeFull(select) else select

      if (vp.tpt.tpe.isRepeatedParam)
        tpd.SeqLiteral(List(cloned), cloned)
      else cloned
    }))
  }

  private def makeArray(values: List[Tree])(using Context): Tree = {
    val elemTpe = defn.AnyType
    val seqApply = Select(ref(defn.SeqModule.termRef), nme.apply)
    val typedApply = TypeApply(seqApply, List(TypeTree(elemTpe)))
    Apply(typedApply, List(SeqLiteral(values, TypeTree(elemTpe))))
  }

  /** Creates the tuple containing the given elements */
  def tupleTree(elems: List[Tree])(using Context): Tree = {
    val arity = elems.length
    if arity == 0 then
      ref(defn.EmptyTupleModule)
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

  def generateElements(record: tpd.TypeDef, thiz: tpd.This)(using Context): tpd.DefDef = {
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
        val sel: tpd.Tree = tpd.Select(thisRef, m.termRef)
        tupleTree(List(tpd.Literal(Constant(name)), sel))
    }

    val dataTpe = requiredClassRef("chisel3.Data")
    val rhs = makeArray(currentFields)

    // Create outer accessor symbols manually and clone declarations
    val decls = bundleSym.info.decls
    val newDecls = decls.cloneScope

    // Mimic newOuterAccessors behavior
    val outerName = nme.OUTER
    val outerFlags = Flags.Method | Flags.Synthetic | Flags.Artifact
    val outerSym = newSymbol(bundleSym, outerName, outerFlags, bundleSym.owner.thisType)
    newDecls.enter(outerSym)

    val updatedInfo = bundleSym.info match {
      case classInfo: ClassInfo =>
        classInfo.derivedClassInfo(decls = newDecls)
      case other =>
        report.error("Expected ClassInfo for bundleSym"); other
    }
    // bundleSym.updateInfo(updatedInfo)

    val elementsSym: Symbol = {
      newSymbol(
        bundleSym,
        Names.termName("_elementsImpl"),
        Flags.Method | Flags.Override | Flags.Protected,
        MethodType(Nil, Nil, defn.AnyType),
      )
    }
    println(s"OUTER: ${elementsSym.info.decl(nme.OUTER).symbol}")

    val dd = tpd.DefDef(elementsSym.asTerm, rhs)
    println(s"dd: $dd")
    dd
  }
}

class BundleComponent extends StandardPlugin {
  val name:                 String = "BundleComponent"
  override val description: String = "Bundle handling"

  override def init(options: List[String]): List[PluginPhase] = {
    (new BundleComponentPhase) :: Nil
  }
}

class BundleComponentPhase extends PluginPhase {
  val phaseName: String = "bundleComponentPhase"
  override val runsAfter = Set(TyperPhase.name)

  override def transformTypeDef(record: tpd.TypeDef)(using Context): tpd.Tree = {
    println("running bundle")
    val k = ChiselTypeHelpers.isRecord(record.tpe) && !record.symbol.flags.is(Flags.Abstract)
    println(s"ChiselTypeHelpers.isRecord(record.tpe): ${ChiselTypeHelpers.isRecord(record.tpe)}")
    println(s"record.symbol.flags.is(Flags.Abstract): ${record.symbol.flags.is(Flags.Abstract)}")
    println(s"entering this $k")
    println(s"record tpe: ${record.tpe}")
    if (ChiselTypeHelpers.isRecord(record.tpe)
      && !record.symbol.flags.is(Flags.Abstract)) {

      val isBundle: Boolean = ChiselTypeHelpers.isBundle(record.tpe)
      val thiz: tpd.This = tpd.This(record.symbol.asClass)

      // todo: test this after genElements
      // // ==================== Generate _cloneTypeImpl ====================
      // val conArgs: Option[List[List[tpd.Tree]]] = BundleHelpers.extractConArgs(record, thiz, isBundle)
      // val cloneTypeImplOpt = BundleHelpers.generateAutoCloneType(record, thiz, conArgs, isBundle)

      println(s"record: $record")
      println(s"isBundle: $isBundle")
      // ==================== Generate val elements (Bundles only) ====================
      val elementsImplOpt: Option[tpd.DefDef] =
        if (isBundle) Some(BundleHelpers.generateElements(record, thiz)) else None

      // ==================== Generate _usingPlugin ====================
      val usingPluginOpt =
        if (isBundle) {
          val isPluginSym = newSymbol(
            record.symbol.owner,
            Names.termName("_usingPlugin"),
            Flags.Method | Flags.Override | Flags.Protected,
            defn.BooleanType
          )
          Some(tpd.DefDef(
            isPluginSym.asTerm,
            _ => tpd.Literal(Constant(true))
          ))
        } else None

      // todo
      // val autoTypenameOpt =
      //   if (BundleHelpers.isAutoTypenamed(record.symbol)) {
      //     BundleHelpers.generateAutoTypename(record, thiz, conArgs.map(_.flatten))
      //   } else None

      val ret = record match {
        case td @ TypeDef(name, tmpl: tpd.Template) => {
          val newDefs = elementsImplOpt.toList ++ usingPluginOpt.toList
          val newTemplate =
            if (tmpl.body.size >= 1)
              cpy.Template(tmpl)(body = newDefs ++ tmpl.body)
            else
              cpy.Template(tmpl)(body = newDefs)
          tpd.cpy.TypeDef(td)(name, newTemplate)
        }
        case _ => super.transformTypeDef(record)
      }
      ret
    } else {
      super.transformTypeDef(record)
    }
  }
}
