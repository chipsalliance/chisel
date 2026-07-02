package unroll

import dotty.tools.dotc.*
import plugins.*
import core.*
import Contexts.*
import Symbols.*
import Flags.*
import SymDenotations.*
import Decorators.*
import ast.Trees.*
import ast.tpd
import StdNames.nme
import Names.*
import Constants.Constant
import dotty.tools.dotc.core.NameKinds.DefaultGetterName
import dotty.tools.dotc.core.Types.{MethodType, NamedType, PolyType, Type}
import dotty.tools.dotc.core.Symbols

import scala.language.implicitConversions
import dotty.tools.dotc.util.SrcPos

class UnrollPhaseScala3() extends PluginPhase {
  import tpd._

  val phaseName = "unroll2"

  override val runsAfter = Set(transform.Pickler.name)

  def copyParam(p: ValDef, parent: Symbol)(using Context) = {
    implicitly[Context].typeAssigner.assignType(
      cpy.ValDef(p)(p.name, p.tpt, p.rhs),
      Symbols.newSymbol(parent, p.name, p.symbol.flags, p.symbol.info)
    )
  }

  def copyParam2(p: TypeDef, parent: Symbol)(using Context) = {
    implicitly[Context].typeAssigner.assignType(
      cpy.TypeDef(p)(p.name, p.rhs),
      Symbols.newSymbol(parent, p.name, p.symbol.flags, p.symbol.info)
    )
  }

  /**
   * Adapted from:
   * - https://github.com/scala/scala3/pull/21693/files#diff-367e32065f51ed46353fdaaf526ab7e7899404b219d24d5b1054fce7f376dfe5R127
   * - https://github.com/scala/scala3/pull/21693/files#diff-e054695755ff26925ae51361df0f7cd4940bc1fd7ceb658023d0dc38c18178c3R3412
   * - https://github.com/scala/scala3/pull/22926/files#diff-367e32065f51ed46353fdaaf526ab7e7899404b219d24d5b1054fce7f376dfe5R135
   */
  private def isValidUnrolledMethod(method: Symbol, origin: SrcPos)(using Context) = {
    val isCtor = method.isConstructor

    def explanation =
      def what = if isCtor then i"a ${if method.owner.is(Trait) then "trait" else "class"} constructor" else i"method ${method.name}"
      val prefix = s"Cannot unroll parameters of $what"
      if method.isLocal then
        i"$prefix because it is a local method"
      else if !method.isEffectivelyFinal then
        i"$prefix because it can be overridden"
      else if isCtor && method.owner.is(Trait) then
        i"implementation restriction: $prefix"
      else if method.owner.companionClass.is(CaseClass) then
        i"$prefix of a case class companion object: please annotate the class constructor instead"
      else
        i"$prefix of a case class: please annotate the class constructor instead"

    if method.name.is(DefaultGetterName) then false
    else if method.isLocal
      || !method.isEffectivelyFinal
      || isCtor && method.owner.is(Trait)
      || method.owner.companionClass.is(CaseClass) && (method.name == nme.apply || method.name == nme.fromProduct)
      || method.owner.is(CaseClass) && method.name == nme.copy then
        report.error(explanation, origin)
        false
    else true
  }


  // if `annotatedMethod` is a case class apply, case class copy or case class fromProduct then `params` actually comes from the case class' primary constructor
  // but we still need to check the actual annotated method hence the two-step check
  def findUnrollAnnotations(params: List[Symbol], specialCasedMethod: Option[Symbol])(using Context): List[Int] = {

    specialCasedMethod.foreach { annotatedMethod => 
      if methodHasUnroll(annotatedMethod) then isValidUnrolledMethod(annotatedMethod, annotatedMethod.sourcePos)
    }
      
    params
      .zipWithIndex
      .collect {
        case (v, i) if hasUnrollAnnotation(v) && isValidUnrolledMethod(v.owner, v.sourcePos) =>
          i
      }
  }

  private def methodHasUnroll(method: Symbol)(using Context) =
    method.paramSymss.exists(_.exists(hasUnrollAnnotation))

  private def hasUnrollAnnotation(sym: Symbol)(using Context) = 
    sym.annotations.exists(_.symbol.fullName.toString == "com.lihaoyi.unroll")

  def isTypeClause(p: ParamClause) = p.headOption.exists(_.isInstanceOf[TypeDef])
  def generateSingleForwarder(defdef: DefDef,
                              prevMethodType: Type,
                              paramIndex: Int,
                              nextParamIndex: Int,
                              nextSymbol: Symbol,
                              annotatedParamListIndex: Int,
                              paramLists: List[ParamClause],
                              isCaseApply: Boolean)
                             (using Context) = {

    def truncateMethodType0(tpe: Type, n: Int): Type = {
      tpe match{
        case pt: PolyType => PolyType(pt.paramNames, pt.paramInfos, truncateMethodType0(pt.resType, n + 1))
        case mt: MethodType =>
          if (n == annotatedParamListIndex) MethodType(mt.paramInfos.take(paramIndex), mt.resType)
          else MethodType(mt.paramInfos, truncateMethodType0(mt.resType, n + 1))
      }
    }

    val truncatedMethodType = truncateMethodType0(prevMethodType, 0)
    val forwarderDefSymbol = Symbols.newSymbol(
      defdef.symbol.owner,
      defdef.name,
      defdef.symbol.flags &~
      HasDefaultParams &~
      (if (nextParamIndex == -1) Flags.EmptyFlags else Deferred) |
      Invisible,
      truncatedMethodType
    )

    val newParamLists: List[ParamClause] = paramLists.zipWithIndex.map{ case (ps, i) =>
      if (i == annotatedParamListIndex) ps.take(paramIndex).map(p => copyParam(p.asInstanceOf[ValDef], forwarderDefSymbol))
      else {
        if (isTypeClause(ps)) ps.map(p => copyParam2(p.asInstanceOf[TypeDef], forwarderDefSymbol))
        else ps.map(p => copyParam(p.asInstanceOf[ValDef], forwarderDefSymbol))
      }
    }

    val defaultOffset = paramLists
      .iterator
      .take(annotatedParamListIndex)
      .filter(!isTypeClause(_))
      .map(_.size)
      .sum

    val defaultCalls = Range(paramIndex, nextParamIndex).map(n =>
      val inner = if (defdef.symbol.isConstructor) {
        ref(defdef.symbol.owner.companionModule)
          .select(DefaultGetterName(defdef.name, n + defaultOffset))
      } else if (isCaseApply) {
        ref(defdef.symbol.owner.companionModule)
          .select(DefaultGetterName(termName("<init>"), n + defaultOffset))
      } else {
        This(defdef.symbol.owner.asClass)
          .select(DefaultGetterName(defdef.name, n + defaultOffset))
      }

      newParamLists
        .take(annotatedParamListIndex)
        .map(_.map(p => ref(p.symbol)))
        .foldLeft[Tree](inner){
          case (lhs: Tree, newParams) =>
            if (newParams.headOption.exists(_.isInstanceOf[TypeTree])) TypeApply(lhs, newParams)
            else Apply(lhs, newParams)
        }
    )

    val forwarderInner: Tree = This(defdef.symbol.owner.asClass).select(nextSymbol)

    val forwarderCallArgs =
      newParamLists.zipWithIndex.map{case (ps, i) =>
        if (i == annotatedParamListIndex) ps.map(p => ref(p.symbol)).take(nextParamIndex) ++ defaultCalls
        else ps.map(p => ref(p.symbol))
      }

    lazy val forwarderCall0 = forwarderCallArgs.foldLeft[Tree](forwarderInner){
      case (lhs: Tree, newParams) =>
        if (newParams.headOption.exists(_.isInstanceOf[TypeTree])) TypeApply(lhs, newParams)
        else Apply(lhs, newParams)
    }

    lazy val forwarderCall =
      if (!defdef.symbol.isConstructor) forwarderCall0
      else Block(List(forwarderCall0), Literal(Constant(())))

    val forwarderDef = implicitly[Context].typeAssigner.assignType(
      cpy.DefDef(defdef)(
        name = forwarderDefSymbol.name,
        paramss = newParamLists,
        tpt = defdef.tpt,
        rhs = if (nextParamIndex == -1) EmptyTree else forwarderCall
      ),
      forwarderDefSymbol
    )

    forwarderDef
  }

  def generateFromProduct(startParamIndices: List[Int], paramCount: Int, defdef: DefDef)(using Context) = {
    // Handle both Block(stmts, Apply(...)) and direct Apply(...) forms
    // In some Scala 3 versions, the rhs is a Block, in others it's directly an Apply
    val (stmts, select, args) = defdef.rhs match {
      case Block(stmts, Apply(select, args)) => (stmts, select, args)
      case Apply(select, args) => (Nil, select, args)
    }
    cpy.DefDef(defdef)(
      name = defdef.name,
      paramss = defdef.paramss,
      tpt = defdef.tpt,
      rhs = Match(
        ref(defdef.paramss.head.head.asInstanceOf[ValDef].symbol).select(termName("productArity")),
        startParamIndices.map { paramIndex =>
          CaseDef(
            Literal(Constant(paramIndex)),
            EmptyTree,
            Block(
              stmts.take(paramIndex),
                Apply(
                select,
                args.take(paramIndex) ++
                  Range(paramIndex, paramCount).map(n =>
                    ref(defdef.symbol.owner.companionModule)
                      .select(DefaultGetterName(defdef.symbol.owner.primaryConstructor.name.toTermName, n))
                  )
              )
            )
          )
        } ++ Seq(
          CaseDef(
            EmptyTree,
            EmptyTree,
            defdef.rhs
          )
        )
      )
    ).setDefTree
  }

  def generateSyntheticDefs(tree: Tree)(using Context): (Option[Symbol], Seq[Tree]) = tree match{
    case defdef: DefDef if defdef.paramss.nonEmpty =>
      import dotty.tools.dotc.core.NameOps.isConstructorName

      LintInnerMethods.traverseChildren(defdef.rhs)

      val isCaseCopy =
        defdef.name.toString == "copy" && defdef.symbol.owner.is(CaseClass)

      val isCaseApply =
        defdef.name.toString == "apply" && defdef.symbol.owner.companionClass.is(CaseClass)

      val isCaseFromProduct = defdef.name.toString == "fromProduct" && defdef.symbol.owner.companionClass.is(CaseClass)

      val annotated =
        if (isCaseCopy) defdef.symbol.owner.primaryConstructor
        else if (isCaseApply) defdef.symbol.owner.companionClass.primaryConstructor
        else if (isCaseFromProduct) defdef.symbol.owner.companionClass.primaryConstructor
        else defdef.symbol

      val specialCasedMethod = Option.when(isCaseCopy || isCaseApply || isCaseFromProduct)(defdef.symbol)

      annotated
        .paramSymss
        .zipWithIndex
        .flatMap{case (paramClause, paramClauseIndex) =>
          val annotationIndices = findUnrollAnnotations(paramClause, specialCasedMethod)
          if (annotationIndices.isEmpty) None
          else Some((paramClauseIndex, annotationIndices))
        }  match{
        case Nil => (None, Nil)
        case Seq((paramClauseIndex, annotationIndices)) =>
          val paramCount = annotated.paramSymss(paramClauseIndex).size
          if (isCaseFromProduct) {
            (Some(defdef.symbol), Seq(generateFromProduct(annotationIndices, paramCount, defdef)))
          } else {
            if (defdef.symbol.is(Deferred)){
              (
                Some(defdef.symbol),
                (-1 +: annotationIndices :+ paramCount).sliding(2).toList.foldLeft((Seq.empty[DefDef], defdef.symbol)) {
                  case ((defdefs, nextSymbol), Seq(paramIndex, nextParamIndex)) =>
                    val forwarder = generateSingleForwarder(
                      defdef,
                      defdef.symbol.info,
                      nextParamIndex,
                      paramIndex,
                      nextSymbol,
                      paramClauseIndex,
                      defdef.paramss,
                      isCaseApply
                    )
                    (forwarder +: defdefs, forwarder.symbol)
                }._1
              )

            }else{

              (
                None,
                (annotationIndices :+ paramCount).sliding(2).toList.reverse.foldLeft((Seq.empty[DefDef], defdef.symbol)){
                  case ((defdefs, nextSymbol), Seq(paramIndex, nextParamIndex)) =>
                    val forwarder = generateSingleForwarder(
                      defdef,
                      defdef.symbol.info,
                      paramIndex,
                      nextParamIndex,
                      nextSymbol,
                      paramClauseIndex,
                      defdef.paramss,
                      isCaseApply
                    )
                    (forwarder +: defdefs, forwarder.symbol)
                }._1
              )
            }
          }

        case multiple => sys.error("Cannot have multiple parameter lists containing `@unroll` annotation")
      }

    case _ => (None, Nil)
  }

  private object LintInnerMethods extends TreeTraverser {
    override def traverse(tree: Tree)(using Context): Unit =
      tree match {
        case method: DefDef =>
          if methodHasUnroll(method.symbol) then isValidUnrolledMethod(method.symbol, method.sourcePos)
          traverseChildren(method.rhs)
        case _ => ()
      }
    override def traverseChildren(tree: Tree)(using Context): Unit = super.traverseChildren(tree)
  }

  override def transformTemplate(tmpl: tpd.Template)(using Context): tpd.Tree = {

    val (removed0, generatedDefs) = tmpl.body.map(generateSyntheticDefs).unzip
    val (None, generatedConstr) = generateSyntheticDefs(tmpl.constr)
    val removed = removed0.flatten

    super.transformTemplate(
      cpy.Template(tmpl)(
        tmpl.constr,
        tmpl.parents,
        tmpl.derived,
        tmpl.self,
        tmpl.body.filter(t => !removed.contains(t.symbol)) ++ generatedDefs.flatten ++ generatedConstr
      )
    )
  }
}
