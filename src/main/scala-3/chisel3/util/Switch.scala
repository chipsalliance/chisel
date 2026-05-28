// SPDX-License-Identifier: Apache-2.0

/** Conditional blocks.
  */

package chisel3.util

import chisel3.*
import scala.quoted.*

/** Conditional logic to form a switch block. See [[is$ is]] for the case API.
  *
  * @example {{{
  * switch (myState) {
  *   is (state1) {
  *     // some logic here that runs when myState === state1
  *   }
  *   is (state2) {
  *     // some logic here that runs when myState === state2
  *   }
  * }
  * }}}
  */
object switch {
  inline def apply[T <: Element](cond: T)(inline x: Any): Unit =
    ${ SwitchMacros.impl[T]('cond, 'x) }
}

private object SwitchMacros {

  def impl[T <: Element: Type](cond: Expr[T], x: Expr[Any])(using Quotes): Expr[Unit] = {
    import quotes.reflect.*

    val isApplySymbol = Symbol.requiredModule("chisel3.util.is").methodMember("apply").toSet
    // Since switch.apply is inlined it will always be wrapped in an Inline block
    val statements: List[Statement] = x.asTerm match {
      case Inlined(_, _, Block(head, tail)) => head :+ tail
    }

    // List of params and blocks as in `is(params) { block }`. The
    // parameter list may have one Iterable arg, one Element arg, or
    // multiple Element args (the `is(v: T, vr: T*)` overload).
    val isCallExprs: List[Expr[(Iterable[T], () => Any)]] = statements.flatMap {
      case Apply(Apply(fun, paramArgs), List(blockArg)) if isApplySymbol(fun.symbol) =>
        Some(buildIsCallExpr[T](paramArgs, blockArg))

      case term =>
        report.errorAndAbort(
          s"Cannot include blocks that do not begin with is() in switch. Got: ${term.show}",
          term.pos
        )
    }

    if (isCallExprs.isEmpty) {
      '{ new SwitchContext[T]($cond, None, Set.empty); () }
    } else {
      val isCallsList: Expr[List[(Iterable[T], () => Any)]] = Expr.ofList(isCallExprs)
      '{
        val ctx = new SwitchContext[T]($cond, None, Set.empty)
        $isCallsList.foldLeft(ctx) { case (acc, (params, block)) =>
          acc.is(params)(block())
        }
        ()
      }
    }
  }

  private def buildIsCallExpr[T <: Element: Type](
    using Quotes
  )(paramArgs: List[quotes.reflect.Term], blockArg: quotes.reflect.Term): Expr[(Iterable[T], () => Any)] = {
    import quotes.reflect.*

    // Unwrap any vararg wrappers added by the typer (`Typed(SeqLiteral(...), Repeated)`).
    val flatArgs: List[Term] = paramArgs.flatMap {
      case Typed(Repeated(elems, _), _) => elems
      case Repeated(elems, _)           => elems
      case other                        => List(other)
    }

    val paramsExpr: Expr[Iterable[T]] = flatArgs match {
      case List(single) =>
        single.asExpr match {
          case '{ $iter: Iterable[T] } => iter
          case '{ $elem: T }           => '{ Seq($elem) }
          case other =>
            report.errorAndAbort(s"is() parameter must be an Element or Iterable[Element], got: ${other.show}")
        }
      case multi =>
        val elemExprs = multi.map(_.asExprOf[T])
        '{ Seq(${ Varargs(elemExprs) }*) }
    }

    val blockExpr: Expr[() => Any] = blockArg.asExpr match {
      case '{ $b: t } => '{ () => $b }
    }

    '{ ($paramsExpr, $blockExpr) }
  }
}
