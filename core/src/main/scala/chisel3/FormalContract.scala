// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._

/** Create a `contract` block.
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
  * The following is a contract with no arguments.
  * {{{
  * FormalContract {
  *   RequireProperty(a >= 1)
  *   EnsureProperty(a + a >= 2)
  * }
  * }}}
  *
  * The following is a contract with a single argument. It expresses the fact
  * that the hardware implementation `(a << 3) + a` is equivalent to `a * 9`.
  * During formal verification, this contract is interpreted as:
  *
  * - `assert((a << 3) + a === a * 9)` as a proof that the contract holds.
  * - `assume(b === a * 9)` on a new symbolic value `b` as a simplification of
  *   other proofs.
  *
  * {{{
  * val b = FormalContract((a << 3) + a) { b =>
  *   EnsureProperty(b === a * 9)
  * }
  * }}}
  *
  * The following is a contract with multiple arguments. It expresses the fact
  * that a carry-save adder with `p`, `q`, and `r` as inputs reduces the number
  * of sum terms from `p + q + r` to `c + s` but doesn't change the sum of the
  * terms. During formal verification, this contract is interpreted as:
  *
  * - `assert(c + s === p + q + r)` as a proof that the carry-save adder indeed
  *   compresses the three terms down to only two terms which have the same sum.
  * - `assume(u + v === p + q + r)` on new symbolic values `u` and `v` as a
  *   simplification of other proofs.
  *
  * {{{
  * val s = p ^ q ^ r
  * val c = (p & q | (p ^ q) & r) << 1
  * val (u, v) = FormalContract(c, s) { case (u, v) =>
  *   EnsureProperty(u + v === p + q + r)
  * }
  * }}}
  */
object FormalContract extends FormalContract$Intf {

  /** Create a formal contract from a sequence of expressions.
    *
    * The `mapping` function is used to convert the `Seq[Data]` to a
    * user-defined type `R`, which is then passed to the contract body and
    * returned as a result.
    *
    * This function is mainly intended for internal use, where the `mapping`
    * function can be used to map the args sequence to a tuple that is more
    * convenient for a user to work with.
    *
    * @example
    * Example with identity mapping. The `Seq(a, b)` is passed to `withSeqArgs`
    * unmodified, and the contract returns a `Seq[Data]` result.
    * {{{
    * def withSeqArgs(args: Seq[Data]): Unit = ...
    * val seqResult: Seq[Data] =
    *   FormalContract.mapping(Seq(a, b), _ => _)(withSeqArgs)
    * }}}
    *
    * Example with a mapping from sequence to a tuple. The `Seq(a, b)` is mapped
    * to a `(UInt, UInt)` tuple through `mapToTuple`, which is then passed to
    * `withTupleArgs`, and the contract returns a corresponding `(UInt, UInt)`
    * result.
    * {{{
    * def mapToTuple(args: Seq[Data]): (UInt, UInt) = {
    *   // check number of args and types, convert to tuple
    * }
    * def withTupleArgs(args: (UInt, UInt)): Unit = ...
    * val tupleResult: (UInt, UInt) =
    *   FormalContract.mapping(Seq(a, b), mapToTuple)(withTupleArgs)
    * }}}
    */
  def mapped[R](args: Seq[Data], mapping: Seq[Data] => R)(
    body: R => Unit
  )(implicit sourceInfo: SourceInfo): R = {
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

  /** Create a `contract` block with no arguments and results. */
  private[chisel3] def _applyNoArgsImpl(body: => Unit)(implicit sourceInfo: SourceInfo): Unit = {
    mapped[Unit](Seq(), _ => ())(_ => body)
  }
}
