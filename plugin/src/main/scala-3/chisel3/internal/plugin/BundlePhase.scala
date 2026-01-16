// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import dotty.tools.dotc.report
import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.ast.tpd.TreeOps
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Symbols.*
import dotty.tools.dotc.core.Names
import dotty.tools.dotc.core.StdNames.nme
import dotty.tools.dotc.core.Constants.Constant
import dotty.tools.dotc.typer.TyperPhase
import dotty.tools.dotc.plugins.PluginPhase
import dotty.tools.dotc.core.Types.*
import dotty.tools.dotc.core.Flags
import dotty.tools.dotc.core.Decorators.toTermName
import dotty.tools.dotc.core.Definitions
import dotty.tools.dotc.transform.PickleQuotes

import scala.collection.mutable

object BundleHelpers {
  def cloneTypeFull(tree: tpd.Tree)(using Context): tpd.Tree = {
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
    record:  tpd.TypeDef,
    thiz:    tpd.This,
    conArgs: List[List[tpd.Tree]]
  )(using Context): tpd.DefDef = {
    val typeParams = record.symbol.typeParams
    val classType = if (typeParams.isEmpty) {
      record.symbol.typeRef
    } else {
      record.symbol.typeRef.appliedTo(typeParams.map(_.typeRef))
    }
    val newExpr: tpd.Tree = conArgs match {
      case Nil => tpd.New(classType, Nil)
      case head :: tail =>
        val headApp = tpd.New(classType, head)
        tail.foldLeft(headApp: tpd.Tree) { (fun, args) =>
          tpd.Apply(fun, args)
        }
    }
    val recordTpe = requiredClassRef("chisel3.Record")
    val cloneTypeSym = newSymbol(
      record.symbol,
      Names.termName("_cloneTypeImpl"),
      Flags.Method | Flags.Override | Flags.Protected,
      MethodType(Nil, Nil, recordTpe)
    )
    tpd.DefDef(cloneTypeSym.asTerm, _ => newExpr)
  }

  // Creates a list of constructor parameter accessors from the
  // argument value. Returns a Some if a constructor is found, or
  // None otherwise.
  def extractConArgs(
    record: tpd.TypeDef,
    thiz:   tpd.This
  )(using Context): Option[List[List[tpd.Tree]]] = {
    val template = record.rhs.asInstanceOf[tpd.Template]
    val primaryConstructorOpt = Option(template.constr)
    val paramAccessors = record.symbol.primaryConstructor.paramSymss.flatten

    if (primaryConstructorOpt.isEmpty) {
      report.warning("Unable to determine primary constructor!", record.sourcePos)
      return None
    }

    val constructor = primaryConstructorOpt.get
    val paramLookup: Map[String, Symbol] =
      paramAccessors.map(sym => sym.name.toString -> sym).toMap

    val symAccessorMap: Map[Names.Name, Symbol] =
      record.symbol.asClass.paramAccessors.map(param => param.name -> param).toMap

    if (constructor.symbol.is(Flags.Private)) {
      val msg = "Private bundle constructors cannot automatically be cloned, try making it package private"
      report.error(msg, constructor.srcPos)
      return None
    }

    val paramSymss = record.symbol.primaryConstructor.paramSymss
      .filterNot(_.exists(_.isType))

    Some(paramSymss.map(_.map { paramSym =>
      // Try to find the accessor in symAccessorMap first (for private
      // fields), otherwise use the param symbol's name directly.
      // Always use Select through `thiz` to properly access the field
      val accessorOpt = symAccessorMap.get(paramSym.name)
      val select = accessorOpt match {
        case Some(accessor) => tpd.Select(thiz, accessor.asTerm.termRef)
        case None           => tpd.Select(thiz, paramSym.name)
      }
      val cloned: tpd.Tree =
        if (ChiselTypeHelpers.isData(paramSym.info))
          cloneTypeFull(select)
        else select
      if (paramSym.info.isRepeatedParam)
        tpd.SeqLiteral(List(cloned), cloned)
      else
        cloned
    }))
  }

  private def makeVector(values: List[tpd.Tree])(using Context): tpd.Tree = {
    val elemTpe = defn.TupleClass.typeRef.appliedTo(List(defn.StringType, defn.AnyType))
    val vectorModule = tpd.ref(requiredModule("scala.collection.immutable.Vector").termRef)

    tpd.Apply(
      tpd.TypeApply(tpd.Select(vectorModule, nme.apply), List(tpd.TypeTree(elemTpe))),
      List(tpd.SeqLiteral(values, tpd.TypeTree(elemTpe)))
    )
  }

  /** Creates the tuple containing the given elements */
  def tupleTree(elems: List[tpd.Tree])(using Context): tpd.Tree = {
    val arity = elems.length
    if arity == 0 then tpd.ref(defn.EmptyTupleModule)
    else if arity <= Definitions.MaxTupleArity then
      // TupleN[elem1Tpe, ...](elem1, ...)
      tpd
        .ref(defn.TupleType(arity).nn.typeSymbol.companionModule)
        .select(nme.apply)
        .appliedToTypes(elems.map(_.tpe.widenIfUnstable))
        .appliedToArgs(elems)
    else
      // TupleXXL.apply(elems*) // TODO add and use Tuple.apply(elems*) ?
      tpd
        .ref(defn.TupleXXLModule)
        .select(nme.apply)
        .appliedToVarargs(elems.map(_.asInstance(defn.ObjectType)), tpd.TypeTree(defn.ObjectType))
        .asInstance(defn.tupleType(elems.map(elem => elem.tpe.widenIfUnstable)))
  }

  def getBundleFields(record: tpd.TypeDef)(using Context): List[tpd.Tree] = {
    val bundleSym = record.symbol.asClass
    val isIgnoreSeq = ChiselTypeHelpers.isIgnoreSeq(record.tpe)

    def isBundleDataField(m: Symbol): Boolean = {
      m.isPublic
      && !m.is(Flags.Method)
      && (
        ChiselTypeHelpers.isData(m.info)
          || ChiselTypeHelpers.isBoxedData(m.info, isIgnoreSeq)
      )
    }

    // Recursively get all bundle fields from this class and its parents
    def getAllBundleFields(sym: ClassSymbol, depth: Int = 0): List[tpd.Tree] = {
      val thisRef: tpd.Tree = tpd.This(bundleSym)

      val currentFields: List[tpd.Tree] = sym.info.decls.toList.collect {
        case m if isBundleDataField(m) =>
          val name = m.name.show
          val memberInBundle = bundleSym.info.member(m.name)
          val sel: tpd.Tree = tpd.Select(thisRef, memberInBundle.symbol.asTerm.termRef)
          tupleTree(List(tpd.Literal(Constant(name)), sel))
      }

      val parentFields: List[tpd.Tree] = if (!ChiselTypeHelpers.isExactBundle(sym)) {
        sym.info.parents.flatMap { parentTpe =>
          parentTpe.classSymbol match {
            case parentSym: ClassSymbol
                if !ChiselTypeHelpers.isExactBundle(parentSym) && ChiselTypeHelpers.isBundle(parentTpe) =>
              getAllBundleFields(parentSym, depth + 1)
            case _ => Nil
          }
        }
      } else {
        Nil
      }
      parentFields ++ currentFields
    }

    getAllBundleFields(bundleSym)
  }

  def generateElements(record: tpd.TypeDef)(using Context): tpd.DefDef = {
    val bundleSym = record.symbol.asClass
    val currentFields: List[tpd.Tree] = getBundleFields(record)
    val dataTpe = requiredClassRef("chisel3.Data")
    val rhs = makeVector(currentFields)

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
  override val runsAfter = Set(PickleQuotes.name)

  override def transformTypeDef(record: tpd.TypeDef)(using Context): tpd.Tree = {
    if (
      ChiselTypeHelpers.isRecord(record.tpe)
      && record.isClassDef
      && !record.symbol.flags.is(Flags.Abstract)
    ) {
      val isBundle: Boolean = ChiselTypeHelpers.isBundle(record.tpe)
      val thiz:     tpd.This = tpd.This(record.symbol.asClass)

      // =========== Generate _cloneTypeImpl ==================
      val conArgsOpt: Option[List[List[tpd.Tree]]] =
        BundleHelpers.extractConArgs(record, thiz)
      val cloneTypeImplOpt: Option[tpd.DefDef] = Option.when(!conArgsOpt.isEmpty) {
        BundleHelpers.generateAutoCloneType(record, thiz, conArgsOpt.get)
      }

      // =========== Generate val elements (Bundles only) =====
      val elementsImplOpt: Option[tpd.DefDef] = Option.when(isBundle) {
        BundleHelpers.generateElements(record)
      }

      // =========== Generate _usingPlugin ====================
      val usingPluginOpt: Option[tpd.DefDef] = Option.when(isBundle) {
        val isPluginSym = newSymbol(
          record.symbol,
          Names.termName("_usingPlugin"),
          Flags.Method | Flags.Override | Flags.Protected,
          defn.BooleanType
        )
        tpd.DefDef(
          isPluginSym.asTerm,
          _ => tpd.Literal(Constant(true))
        )
      }

      record match {
        case td @ tpd.TypeDef(name, tmpl: tpd.Template) => {
          val newDefs = elementsImplOpt ++: usingPluginOpt ++: cloneTypeImplOpt.toList
          val newTemplate =
            if (tmpl.body.size >= 1)
              cpy.Template(tmpl)(body = newDefs ++: tmpl.body)
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
