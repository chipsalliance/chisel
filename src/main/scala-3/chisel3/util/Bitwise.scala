// SPDX-License-Identifier: Apache-2.0

/** Miscellaneous circuit generators operating on bits.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

/** Creates repetitions of each bit of the input in order.
  *
  * @example {{{
  * FillInterleaved(2, "b1 0 0 0".U)  // equivalent to "b11 00 00 00".U
  * FillInterleaved(2, "b1 0 0 1".U)  // equivalent to "b11 00 00 11".U
  * FillInterleaved(2, myUIntWire)  // dynamic interleaved fill
  *
  * FillInterleaved(2, Seq(false.B, false.B, false.B, true.B))  // equivalent to "b11 00 00 00".U
  * FillInterleaved(2, Seq(true.B, false.B, false.B, true.B))  // equivalent to "b11 00 00 11".U
  * }}}
  */
object FillInterleaved extends FillInterleavedImpl {

  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    */
  def apply(n: Int, in: UInt)(using SourceInfo): UInt = _applyImpl(n, in)

  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    */
  def apply(n: Int, in: Seq[Bool])(using SourceInfo): UInt = _applyImpl(n, in)
}

/** Returns the number of bits set (value is 1 or true) in the input signal.
  *
  * @example {{{
  * PopCount(Seq(true.B, false.B, true.B, true.B))  // evaluates to 3.U
  * PopCount(Seq(false.B, false.B, true.B, false.B))  // evaluates to 1.U
  *
  * PopCount("b1011".U)  // evaluates to 3.U
  * PopCount("b0010".U)  // evaluates to 1.U
  * PopCount(myUIntWire)  // dynamic count
  * }}}
  */
object PopCount extends PopCountImpl {

  def apply(in: Iterable[Bool])(using SourceInfo): UInt = _applyImpl(in)
  def apply(in: Bits)(using SourceInfo):           UInt = _applyImpl(in)
}

/** Create repetitions of the input using a tree fanout topology.
  *
  * @example {{{
  * Fill(2, "b1000".U)  // equivalent to "b1000 1000".U
  * Fill(2, "b1001".U)  // equivalent to "b1001 1001".U
  * Fill(2, myUIntWire)  // dynamic fill
  * }}}
  */
object Fill extends FillImpl {

  /** Create n repetitions of x using a tree fanout topology.
    *
    * Output data-equivalent to x ## x ## ... ## x (n repetitions).
    * @throws java.lang.IllegalArgumentException if `n` is less than zero
    */
  def apply(n: Int, x: UInt)(using SourceInfo): UInt = _applyImpl(n, x)
}

/** Returns the input in bit-reversed order. Useful for little/big-endian conversion.
  *
  * @example {{{
  * Reverse("b1101".U)  // equivalent to "b1011".U
  * Reverse("b1101".U(8.W))  // equivalent to "b10110000".U
  * Reverse(myUIntWire)  // dynamic reverse
  * }}}
  */
object Reverse extends ReverseImpl {

  def apply(in: UInt)(using SourceInfo): UInt = _applyImpl(in)
}
