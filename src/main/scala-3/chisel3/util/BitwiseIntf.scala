// SPDX-License-Identifier: Apache-2.0

/** Miscellaneous circuit generators operating on bits.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait FillInterleaved$Intf { self: FillInterleaved.type =>

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

private[chisel3] trait PopCount$Intf { self: PopCount.type =>

  def apply(in: Iterable[Bool])(using SourceInfo): UInt = _applyImpl(in)
  def apply(in: Bits)(using SourceInfo):           UInt = _applyImpl(in)
}

private[chisel3] trait Fill$Intf { self: Fill.type =>

  /** Create n repetitions of x using a tree fanout topology.
    *
    * Output data-equivalent to x ## x ## ... ## x (n repetitions).
    * @throws java.lang.IllegalArgumentException if `n` is less than zero
    */
  def apply(n: Int, x: UInt)(using SourceInfo): UInt = _applyImpl(n, x)
}

private[chisel3] trait Reverse$Intf { self: Reverse.type =>

  def apply(in: UInt)(using SourceInfo): UInt = _applyImpl(in)
}
