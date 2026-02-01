// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import scala.quoted.*

private[chisel3] trait FormalContract$Intf { self: FormalContract.type =>

  /** Create a `contract` block with no arguments and results. */
  def apply(body: => Unit)(using sourceInfo: SourceInfo): Unit = _applyNoArgsImpl(body)

  /** Create a `contract` block with one or more arguments and results. */
  transparent inline def apply[T <: Data](inline args: T*): Any =
    ${ FormalContractMacro('args) }
}

private[chisel3] object FormalContractMacro {

  /** A macro for `FormalContract` that allows for contracts to be built while
    * preserving the types of individual elements passed to the contract. This
    * allows the types of the contract's results to match the type of the
    * arguments passed to the contract.
    *
    * A call to `FormalContract(a: T)` becomes:
    * {{{
    * FormalContract.mapped(
    *   Seq(a),
    *   (values => values(0).asInstanceOf[T])
    * )
    * }}}
    *
    * A call to `FormalContract(a: A, b: B, c: C)` becomes:
    * {{{
    * FormalContract.mapped(
    *   Seq(a, b, c),
    *   (values => (
    *     values(0).asInstanceOf[A],
    *     values(1).asInstanceOf[B],
    *     values(2).asInstanceOf[C]
    *   ))
    * )
    * }}}
    */
  def apply(args: Expr[Seq[Data]])(using Type[Seq[Data]])(using Quotes): Expr[Any] = {
    val elements = args match {
      case Varargs(elements) => elements
      case _                 => quotes.reflect.report.errorAndAbort("Expected a compile-time known list of arguments")
    }
    // In the following we use a `match case '{ _: t}` to bind the type of the
    // left-hand side of the match to `t`, which we can then refer to within the
    // match to cast elements to the appropriate type. This allows us to get the
    // type of the element if there's only one, or we construct a tuple out of
    // all the elements as a way to construct the corresponding tuple type.
    if (elements.size == 0) {
      quotes.reflect.report.errorAndAbort("use `FormalContract` without `()` instead")
    } else if (elements.size == 1) {
      // Calls of the form `apply(x: T)` require a mapping like:
      // ```
      // (values: Seq[Data]) => values.last.asInstanceOf[T]
      // ```
      elements.last match {
        case '{ $x: t } =>
          val mapping = '{ (values: Seq[Data]) => values.last.asInstanceOf[t] }
          '{ FormalContract.mapped(${ Expr.ofSeq(elements) }, $mapping) }
      }
    } else {
      // Calls of the form `apply(x0: T0, x1: T1, ..., xi: Ti)` require a
      // mapping like:
      // ```
      // (values: Seq[Data]) => (
      //   values(0).asInstanceOf[T0],
      //   values(1).asInstanceOf[T1],
      //   ...,
      //   values(i).asInstanceOf[Ti],
      // )
      // ```
      Expr.ofTupleFromSeq(elements) match {
        case '{ $x: t } =>
          val mapping = '{ (values: Seq[Data]) =>
            ${
              Expr.ofTupleFromSeq((0 until elements.size).map(i => '{ values(${ Expr(i) }) }))
            }.asInstanceOf[t]
          }
          '{ FormalContract.mapped(${ Expr.ofSeq(elements) }, $mapping) }
      }
    }
  }
}
