// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.{BaseModule, SourceInfo}
import scala.quoted.*

private[chisel3] trait InstantiateIntf { self: Instantiate.type =>

  /** Create an `Instance` of a `Module`
    *
    * This is similar to `Module(...)` except that it returns an `Instance[_]` object.
    *
    * @param con module construction, must be actual call to constructor (`new MyModule(...)`)
    * @return constructed module `Instance`
    */
  inline def apply[A <: BaseModule](inline con: A): Instance[A] = ${ InstantiateIntfMacros.instanceMacro[A]('con) }

  inline def definition[A <: BaseModule](inline con: A): Definition[A] = ${
    InstantiateIntfMacros.definitionMacro[A]('con)
  }

  def _instance[K, A <: BaseModule](
    args: K,
    f:    K => A,
    tt:   AnyRef
  )(
    using sourceInfo: SourceInfo
  ): Instance[A] = {
    _instanceImpl(args, f, tt)
  }

  /** Internal method for creating a Definition from extracted arguments
    * This is not part of the public API, do not call directly!
    */
  def _definition[K, A <: BaseModule](
    args: K,
    f:    K => A,
    tt:   AnyRef
  ): Definition[A] = _definitionImpl(args, f, tt)
}

private object InstantiateIntfMacros {
  import scala.quoted.*

  // This helper is called by both instanceMacro and definitionMacro.
  // Extracts all args to the Module and creates a constructor lambda
  // of shape `argsTuple() => new Module(...)`
  private def extractAndBuild[A <: BaseModule: Type](
    con: Expr[A]
  )(using q: Quotes): (Expr[Any], Expr[_]) = {
    import q.reflect.*
    // Since our Instantiate apply method is Inline, Scala wraps it in
    // an `Inline` block to track bindings. We need to peel the Inline
    // block to get to the Apply block.
    def unwrapInlined(term: Term): Term = term match {
      case Inlined(_, _, expansion) => unwrapInlined(expansion)
      case other                    => other
    }

    def collectArgs(
      term:          Term,
      accFlat:       List[Term],
      accStructured: List[List[Term]]
    ): (Term, List[Term], List[List[Term]]) = term match {
      case Apply(inner, newArgs) =>
        val isImplicit = inner.tpe.widen match {
          case mt: MethodType => mt.isImplicit
          case _ => false
        }
        if (isImplicit) collectArgs(inner, accFlat, accStructured)
        else collectArgs(inner, newArgs ::: accFlat, newArgs :: accStructured)
      case other => (other, accFlat, accStructured)
    }

    val unwrapped = unwrapInlined(con.asTerm)
    val (core, args, argStructure) = collectArgs(unwrapped, Nil, Nil)

    // Unwrap based on whether this is a constructor with or without
    // explicit type arguments; for example, `MyModule[UInt].apply(a)`
    // is wrapped in a TypeApply block
    val (tpt, typeArgs) = core match {
      case Select(New(tpt), _)                   => (tpt, None)
      case TypeApply(Select(New(tpt), _), targs) => (tpt, Some(targs))
      case _ =>
        report.errorAndAbort(s"Invalid arguments: ${con.show}", con.asTerm.pos)
    }

    // Rebuild the original constructor without arguments
    val constructorBase: Term = typeArgs match {
      case Some(targs) => TypeApply(Select(New(tpt), tpt.tpe.typeSymbol.primaryConstructor), targs)
      case None        => Select(New(tpt), tpt.tpe.typeSymbol.primaryConstructor)
    }

    val argExprs = args.map(_.asExpr)
    val n = argExprs.size

    val argsTuple: Expr[Any] = n match {
      case 0 => '{ () }
      case 1 => argExprs.head
      case _ => Expr.ofTupleFromSeq(argExprs)
    }

    val constructorFunc: Expr[_] = n match {
      case 0 =>
        Lambda(
          Symbol.spliceOwner,
          MethodType(List("x"))(_ => List(TypeRepr.of[Unit]), _ => TypeRepr.of[A]),
          (sym, _) => unwrapped.changeOwner(sym)
        ).asExpr
      case 1 =>
        Lambda(
          Symbol.spliceOwner,
          MethodType(List("arg"))(_ => List(args.head.tpe.widen), _ => TypeRepr.of[A]),
          (sym, params) => {
            val it = Iterator.single(params.head.asInstanceOf[Term])
            val argss = argStructure.map(_.map(_ => it.next()))
            val result = argss.foldLeft(constructorBase)((acc, as) => Apply(acc, as))
            result.asExprOf[A].asTerm.changeOwner(sym)
          }
        ).asExpr
      case _ =>
        val tupleType = AppliedType(defn.TupleClass(n).typeRef, args.map(_.tpe.widen))
        Lambda(
          Symbol.spliceOwner,
          MethodType(List("tupArg"))(_ => List(tupleType), _ => TypeRepr.of[A]),
          (sym, params) => {
            val tup = params.head.asInstanceOf[Term]
            val flatArgs = (1 to n).map(i => Select.unique(tup, s"_$i")).toList
            val it = flatArgs.iterator
            val argss = argStructure.map(_.map(_ => it.next()))
            val result = argss.foldLeft(constructorBase)((acc, as) => Apply(acc, as))
            result.asExprOf[A].asTerm.changeOwner(sym)
          }
        ).asExpr
    }

    (argsTuple, constructorFunc)
  }

  // Transform `Instantiate(new MyModule(arg1, arg2))` into
  // `Instantiate._instance((arg1, arg2), args => new MyModule(args._1, args._2), typeToken)(using sourceInfo)`
  def instanceMacro[A <: BaseModule: Type](con: Expr[A])(using q: Quotes): Expr[Instance[A]] = {
    import q.reflect.*
    val sourceInfoExpr = Expr.summon[SourceInfo].getOrElse {
      report.errorAndAbort("Instantiate.apply requires an implicit SourceInfo")
    }
    val (argsTuple, constructorFunc) = extractAndBuild[A](con)
    val typeRepr = TypeRepr.of[A]
    val typeToken = Expr(typeRepr.show)

    '{
      chisel3.experimental.hierarchy.Instantiate._instance(
        $argsTuple.asInstanceOf[Any],
        $constructorFunc.asInstanceOf[Any => A],
        $typeToken.asInstanceOf[AnyRef]
      )(using $sourceInfoExpr)
    }
  }

  def definitionMacro[A <: BaseModule: Type](con: Expr[A])(using q: Quotes): Expr[core.Definition[A]] = {
    import q.reflect.*
    val (argsTuple, constructorFunc) = extractAndBuild[A](con)
    val typeRepr = TypeRepr.of[A]
    val typeToken = Expr(typeRepr.show)
    '{
      chisel3.experimental.hierarchy.Instantiate._definition(
        $argsTuple.asInstanceOf[Any],
        $constructorFunc.asInstanceOf[Any => A],
        $typeToken.asInstanceOf[AnyRef]
      )
    }
  }
}
