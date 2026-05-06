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
      case app @ Apply(inner, newArgs) =>
        val isImplicit = inner.tpe.widen match {
          case mt: MethodType => mt.isImplicit
          case _ => false
        }

        if (isImplicit) {
          collectArgs(inner, accFlat, accStructured)
        } else {
          collectArgs(inner, newArgs ::: accFlat, newArgs :: accStructured)
        }
      case other => (other, accFlat, accStructured)
    }

    val unwrapped = unwrapInlined(con.asTerm)
    val (core, args, argStructure) = collectArgs(unwrapped, Nil, Nil)

    // Unwrap based on whether this is a constructor with or without
    // explicit type arguments; for example, `MyModule[UInt].apply(a)`
    // is wrapped in a TypeApply block
    val (tpt, typeArgs, fullConstructorTerm) = core match {
      case Select(New(tpt), _) =>
        (tpt, None, unwrapped)
      case TypeApply(Select(New(tpt), _), targs) =>
        (tpt, Some(targs), unwrapped)
      case _ =>
        report.errorAndAbort(
          s"Invalid arguments: ${con.show}",
          con.asTerm.pos
        )
    }

    // Helper to match the original parameter list structure by distributing flat args
    def matchStructure(
      structure: List[List[Term]],
      flatArgs:  List[Term]
    ): List[List[Term]] = {
      val it = flatArgs.iterator
      structure.map(_.map(_ => it.next()))
    }

    def buildConstructorLambdaMultiList(
      paramTypes:              List[TypeRepr],
      reconstructArgStructure: List[Any] => List[List[Term]],
      paramName:               String
    ): Expr[_] = {
      Lambda(
        Symbol.spliceOwner,
        MethodType(List(paramName))(_ => paramTypes, _ => TypeRepr.of[A]),
        (sym, params) => {
          val argss = reconstructArgStructure(params)
          // Build nested Apply nodes for multiple parameter lists
          val base = typeArgs match {
            case Some(targs) => TypeApply(Select(New(tpt), tpt.tpe.typeSymbol.primaryConstructor), targs)
            case None        => Select(New(tpt), tpt.tpe.typeSymbol.primaryConstructor)
          }
          val result = argss.foldLeft(base)((acc, args) => Apply(acc, args))
          result.asExprOf[A].asTerm.changeOwner(sym)
        }
      ).asExpr
    }

    val argExprs = args.map(_.asExpr)

    val argsTuple: Expr[Any] = argExprs.size match {
      case 0 => '{ () }
      case 1 => argExprs.head
      case _ => Expr.ofTupleFromSeq(argExprs)
    }

    val constructorFunc: Expr[_] = argExprs.size match {
      case 0 =>
        Lambda(
          Symbol.spliceOwner,
          MethodType(List("x"))(_ => List(TypeRepr.of[Unit]), _ => TypeRepr.of[A]),
          (sym, _) => fullConstructorTerm.changeOwner(sym)
        ).asExpr
      case 1 =>
        val argTpe = args.head.tpe.widen
        buildConstructorLambdaMultiList(
          List(argTpe),
          params => {
            val singleArg = params.head.asInstanceOf[Term]
            matchStructure(argStructure, List(singleArg))
          },
          "arg"
        )
      case n =>
        val argTypes = args.map(_.tpe.widen)
        val tupleTypeRepr = AppliedType(defn.TupleClass(n).typeRef, argTypes)
        buildConstructorLambdaMultiList(
          List(tupleTypeRepr),
          params => {
            val flatArgs = (1 to n).map(i => Select.unique(params.head.asInstanceOf[Term], s"_$i")).toList
            matchStructure(argStructure, flatArgs)
          },
          "tupArg"
        )
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
