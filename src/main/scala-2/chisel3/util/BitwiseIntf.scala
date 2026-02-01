// SPDX-License-Identifier: Apache-2.0

/** Miscellaneous circuit generators operating on bits.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform
import scala.language.experimental.macros

private[chisel3] trait FillInterleaved$Intf { self: FillInterleaved.type =>

  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    */
  def apply(n: Int, in: UInt): UInt = macro SourceInfoTransform.nInArg

  /** @group SourceInfoTransformMacro */
  def do_apply(n: Int, in: UInt)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(n, in)

  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    */
  def apply(n: Int, in: Seq[Bool]): UInt = macro SourceInfoTransform.nInArg

  /** @group SourceInfoTransformMacro */
  def do_apply(n: Int, in: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt = _applyImpl(n, in)
}

private[chisel3] trait PopCount$Intf { self: PopCount.type =>

  def apply(in: Iterable[Bool]): UInt = macro SourceInfoTransform.inArg

  /** @group SourceInfoTransformMacro */
  def do_apply(in: Iterable[Bool])(implicit sourceInfo: SourceInfo): UInt = _applyImpl(in)

  def apply(in: Bits): UInt = macro SourceInfoTransform.inArg

  /** @group SourceInfoTransformMacro */
  def do_apply(in: Bits)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(in)

  /** Implements PopCount(x)==n with less deep circuitry in case n=0,1,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x UInt to PopCount
    * @return true.B when x has exactly n bits set
    */
  def equalTo(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): Bool = _equalToImpl(n, x)

  /** Implements PopCount(x)==n with less deep circuitry in case n=0,1,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x Seq/Vec of Bool to PopCount
    * @return true.B when x has exactly n bits set
    */
  def equalTo(n: Int, x: Iterable[Bool])(implicit sourceInfo: SourceInfo): Bool = _equalToImpl(n, x)

  /** Implements PopCount(x)>n with less deep circuitry in case n=0,1,x.width-1
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x UInt to PopCount
    * @return true.B when x has more than n bits set
    */
  def greaterThan(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): Bool = _greaterThanImpl(n, x)

  /** Implements PopCount(x)>n with less deep circuitry in case n=0,1,x.width-1
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x Seq/Vec of Bool to PopCount
    * @return true.B when x has more than n bits set
    */
  def greaterThan(n: Int, x: Iterable[Bool])(implicit sourceInfo: SourceInfo): Bool = _greaterThanImpl(n, x)

  /** Implements PopCount(x)>=n with less deep circuitry in case n=0,1,2,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x UInt to PopCount
    * @return true.B when x has n or more bits set
    */
  def atLeast(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): Bool = _atLeastImpl(n, x)

  /** Implements PopCount(x)>=n with less deep circuitry in case n=0,1,2,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x Seq/Vec of Bool to PopCount
    * @return true.B when x has n or more bits set
    */
  def atLeast(n: Int, x: Iterable[Bool])(implicit sourceInfo: SourceInfo): Bool = _atLeastImpl(n, x)
}

private[chisel3] trait Fill$Intf { self: Fill.type =>

  /** Create n repetitions of x using a tree fanout topology.
    *
    * Output data-equivalent to x ## x ## ... ## x (n repetitions).
    * @throws java.lang.IllegalArgumentException if `n` is less than zero
    */
  def apply(n: Int, x: UInt): UInt = macro SourceInfoTransform.nxArg

  /** @group SourceInfoTransformMacro */
  def do_apply(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(n, x)
}

private[chisel3] trait Reverse$Intf { self: Reverse.type =>

  def apply(in: UInt): UInt = macro SourceInfoTransform.inArg

  /** @group SourceInfoTransformMacro */
  def do_apply(in: UInt)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(in)
}
