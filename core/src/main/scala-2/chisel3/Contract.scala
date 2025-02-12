// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._
import scala.language.experimental.macros
import scala.reflect.macros.whitebox

object Contract {

  /** Create a `contract` block with no arguments and results.
    *
    * Within the contract body, `RequireProperty` and `EnsureProperty` can be
    * used to describe the pre and post-conditions of the contract. During
    * formal verification, contracts can be used to divide large formal checks
    * into smaller pieces by first asserting some local properties hold, and
    * then assuming those properties hold in the context of the larger proof.
    *
    * @example
    * {{{
    * Contract() {
    *   RequireProperty(a >= 1)
    *   EnsureProperty(a + a >= 2)
    * }
    * }}}
    */
  def apply()(body: => Unit)(implicit sourceInfo: SourceInfo): Unit = {
    mapped[Unit](Seq(), _ => ())(_ => body)
  }

  /** Create a `contract` block with one or more arguments and results.
    *
    * Outside of verification uses, the arguments passed to the contract are
    * simply returned as contract results. During formal verification, the
    * contract may act as a cutpoint by returning symbolic values for its
    * results and using the contract body as constraint.
    *
    * Within the contract body, `RequireProperty` and `EnsureProperty` can be
    * used to describe the pre and post-conditions of the contract. During
    * formal verification, contracts can be used to divide large formal checks
    * into smaller pieces by first asserting some local properties hold, and
    * then assuming those properties hold in the context of the larger proof.
    *
    * @example
    * {{{
    * // Contract with single argument: "(a<<3)+a is equivalent to a*9".
    * // During formal verification, this is interpreted as:
    * // - "assert((a<<3)+a === a*9)" as a proof that the contract holds
    * // - "assume(b === a*9)" as a simplification of other proofs
    * val b = Contract((a<<3)+a) { b =>
    *   EnsureProperty(b === a*9)
    * }
    *
    * // Contract with multiple arguments:
    * //   "carry-save adder reduces number of terms but doesn't change sum"
    * // During formal verification, this is interpreted as:
    * // - an assertion that the carry-save adder indeed compresses the three
    * //   terms p/q/r down to only two terms u/v which have the same sum
    * // - symbolic values u/v constrained to sum up to the same value as p/q/r,
    * //   as a simplification of other proofs
    * val csa_s = p^q^r
    * val csa_c = (p&q | (p^q)&r) << 1
    * val (u,v) = Contract(csa_c, csa_s) { case (u,v) =>
    *   EnsureProperty(u+v === p+q+r)
    * }
    * }}}
    */
  def apply(head: Data, tail: Data*): (Any => Unit) => Any = macro apply_impl

  // This uses a macro internally to allow for contracts with a single argument
  // and contracts with a tuple of arguments to be created. The macro packages
  // the sequence of arguments up into a tuple (`Tuple1` for single arguments,
  // and the corresponding `TupleN` for multiple arguments). Without this macro,
  // the contracts would always operate on tuples (`Product`), forcing the user
  // to use `val Tuple1(a) = Contract(Tuple1(b)) { case Tuple1(x) => ... }` for
  // contracts with only a single argument.
  def apply_impl(c: whitebox.Context)(head: c.Expr[Data], tail: c.Expr[Data]*): c.Expr[(Any => Unit) => Any] = {
    import c.universe._
    val args = head +: tail
    val mapping =
      q"(values => (..${args.zipWithIndex.map { case (arg, i) => q"values(${i}).asInstanceOf[${arg.tree.tpe}]" }}))"
    val result = q"Contract.mapped(Seq(..${args.map(_.tree)}), $mapping)(_)"
    c.Expr[(Any => Unit) => Any](result)
  }

  /** Create a contract from a sequence of expressions.
    *
    * The `mapping` function is used to convert the `Seq[Data]` to a
    * user-defined type `R`, which is then passed to the contract body and
    * returned as a result.
    */
  def mapped[R](args: Seq[Data], mapping: Seq[Data] => R)(body: R => Unit)(implicit sourceInfo: SourceInfo): R = {
    // Create a sequence of contract results, one for each argument.
    val results = args.map { arg =>
      val result = arg.cloneTypeFull
      experimental.requireIsChiselType(result, "contract result type")
      result.bind(binding.OpBinding(Builder.forcedUserModule, Builder.currentBlock))
      result
    }

    // Create the contract.
    val contract = Builder.pushCommand(new DefContract(sourceInfo, results, args.map(_.ref(sourceInfo))))

    // Map the sequence of results to the user-defined type `R`, pass it to the
    // body to build the contract body, and also return it as the result of the
    // contract as a whole.
    val mapped = mapping(results)
    Builder.forcedUserModule.withRegion(contract.region)(layer.elideBlocks(body(mapped)))
    mapped
  }
}
