// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import scala.language.experimental.macros
import scala.reflect.macros.whitebox

private[chisel3] trait FormalContract$Intf { self: FormalContract.type =>

  /** Create a `contract` block with no arguments and results. */
  def apply(body: => Unit)(implicit sourceInfo: SourceInfo): Unit = _applyNoArgsImpl(body)

  /** Create a `contract` block with one or more arguments and results. */
  def apply(head: Data, tail: Data*): (Any => Unit) => Any =
    macro FormalContractMacro.apply_impl
}

private[chisel3] object FormalContractMacro {

  /** A macro for `FormalContract` to allow for contracts with a single argument
    * and contracts with a tuple of arguments to be created. The macro packages
    * the sequence of arguments up into a tuple (`Tuple1` for single arguments,
    * and the corresponding `TupleN` for multiple arguments). Without this
    * macro, the contracts would always operate on tuples (`Product`), forcing
    * the user to use
    * {{{
    * val Tuple1(a) = FormalContract(Tuple1(b)) { case Tuple1(x) => ... }
    * }}}
    * for contracts with only a single argument.
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
  def apply_impl(c: whitebox.Context)(head: c.Expr[Data], tail: c.Expr[Data]*): c.Expr[(Any => Unit) => Any] = {
    import c.universe._
    val args = head +: tail
    val mapping =
      q"(values => (..${args.zipWithIndex.map { case (arg, i) => q"values(${i}).asInstanceOf[${arg.tree.tpe}]" }}))"
    val result = q"FormalContract.mapped(Seq(..${args.map(_.tree)}), $mapping)(_)"
    c.Expr[(Any => Unit) => Any](result)
  }
}
