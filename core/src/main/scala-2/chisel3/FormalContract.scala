// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import scala.language.experimental.macros
import scala.reflect.macros.whitebox

object FormalContract extends ObjectFormalContractImpl {

  /** Create a `contract` block with one or more arguments and results. */
  def apply(head: Data, tail: Data*): (Any => Unit) => Any = macro apply_impl

  // This uses a macro internally to allow for contracts with a single argument
  // and contracts with a tuple of arguments to be created. The macro packages
  // the sequence of arguments up into a tuple (`Tuple1` for single arguments,
  // and the corresponding `TupleN` for multiple arguments). Without this macro,
  // the contracts would always operate on tuples (`Product`), forcing the user
  // to use
  //
  //   val Tuple1(a) = FormalContract(Tuple1(b)) { case Tuple1(x) => ... }
  //
  // for contracts with only a single argument.
  /** @group Macro
    * @groupprio Macro 1001
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
