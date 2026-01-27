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

private[util] object SwitchMacros {

  def impl[T <: Element: Type](cond: Expr[T], x: Expr[Any])(using Quotes): Expr[Unit] = {
    import quotes.reflect.*

    val statements: List[Statement] = x.asTerm match {
      case Inlined(_, _, block) => extractStatements(block)
      case block                => extractStatements(block)
    }

    val isCallExprs: List[Expr[(Iterable[T], () => Any)]] = statements.flatMap { stmt =>
      stmt match {
        case term: Term => extractIsCall[T](term)
        case _ => None
      }
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

  private def extractStatements(using Quotes)(term: quotes.reflect.Term): List[quotes.reflect.Statement] = {
    import quotes.reflect.*
    term match {
      case Block(stats, expr) =>
        expr match {
          case Literal(UnitConstant()) => stats
          case _                       => stats :+ expr
        }
      case other => List(other)
    }
  }

  private def extractIsCall[T <: Element: Type](using Quotes)(
    term: quotes.reflect.Term
  ): Option[Expr[(Iterable[T], () => Any)]] = {
    import quotes.reflect.*

    term match {
      case Apply(Apply(TypeApply(Select(Ident("is"), "apply"), _), paramArgs), List(blockArg)) =>
        Some(buildIsCallExpr[T](paramArgs, blockArg))

      case Apply(Apply(Select(Ident("is"), "apply"), paramArgs), List(blockArg)) =>
        Some(buildIsCallExpr[T](paramArgs, blockArg))

      case Apply(
            Apply(TypeApply(Select(Select(Select(Ident("chisel3"), "util"), "is"), "apply"), _), paramArgs),
            List(blockArg)
          ) =>
        Some(buildIsCallExpr[T](paramArgs, blockArg))

      case Apply(
            Apply(Select(Select(Select(Ident("chisel3"), "util"), "is"), "apply"), paramArgs),
            List(blockArg)
          ) =>
        Some(buildIsCallExpr[T](paramArgs, blockArg))

      case Apply(Apply(TypeApply(sel @ Select(_, "apply"), _), paramArgs), List(blockArg))
          if sel.symbol.owner.name == "is" =>
        Some(buildIsCallExpr[T](paramArgs, blockArg))

      case Apply(Apply(sel @ Select(_, "apply"), paramArgs), List(blockArg)) if sel.symbol.owner.name == "is" =>
        Some(buildIsCallExpr[T](paramArgs, blockArg))

      case _ =>
        report.errorAndAbort(
          s"Cannot include blocks that do not begin with is() in switch. Got: ${term.show}",
          term.pos
        )
    }
  }

  private def buildIsCallExpr[T <: Element: Type](
    using Quotes
  )(paramArgs: List[quotes.reflect.Term], blockArg: quotes.reflect.Term): Expr[(Iterable[T], () => Any)] = {
    import quotes.reflect.*

    val paramsExpr: Expr[Iterable[T]] = paramArgs match {
      case List(single) =>
        single.asExpr match {
          case '{ $iter: Iterable[T] } => iter
          case '{ $elem: T }           => '{ Seq($elem) }
          case other =>
            report.errorAndAbort(s"is() parameter must be an Element or Iterable[Element], got: ${other.show}")
        }
      case multiple =>
        val elemExprs = multiple.map { arg =>
          arg.asExpr match {
            case '{ $elem: T } => elem
            case other =>
              report.errorAndAbort(s"is() parameter must be an Element, got: ${other.show}")
          }
        }
        val seqExpr = Expr.ofList(elemExprs)
        '{ $seqExpr }
    }

    val blockExpr: Expr[() => Any] = blockArg.asExpr match {
      case '{ $b: t } => '{ () => $b }
    }

    '{ ($paramsExpr, $blockExpr) }
  }
}
