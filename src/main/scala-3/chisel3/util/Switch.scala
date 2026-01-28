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

    // List of params and blocks as in `is(params) { block }`
    val isCallExprs: List[Expr[(Iterable[T], () => Any)]] = statements.flatMap {
      case Apply(Apply(fun, List(paramArg)), List(blockArg)) if isApplySymbol(fun.symbol) =>
        Some(buildIsCallExpr[T](paramArg, blockArg))

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
  )(paramArg: quotes.reflect.Term, blockArg: quotes.reflect.Term): Expr[(Iterable[T], () => Any)] = {
    import quotes.reflect.*

    val paramsExpr: Expr[Iterable[T]] = paramArg.asExpr match {
      case '{ $iter: Iterable[T] } => iter
      case '{ $elem: T }           => '{ Seq($elem) }
      case other =>
        report.errorAndAbort(s"is() parameter must be an Element or Iterable[Element], got: ${other.show}")
    }

    val blockExpr: Expr[() => Any] = blockArg.asExpr match {
      case '{ $b: t } => '{ () => $b }
    }

    '{ ($paramsExpr, $blockExpr) }
  }
}
