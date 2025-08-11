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
object FillInterleaved extends FillInterleaved$Intf {

  protected def _applyImpl(n: Int, in: UInt)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(n, in.asBools)

  protected def _applyImpl(n: Int, in: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt =
    Cat(in.map(Fill(n, _)).reverse)
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
object PopCount extends PopCount$Intf {

  protected def _applyImpl(in: Iterable[Bool])(implicit sourceInfo: SourceInfo): UInt = SeqUtils.count(in.toSeq)

  protected def _applyImpl(in: Bits)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(
    (0 until in.getWidth).map(in(_))
  )

  /** Implements PopCount(x)==n with less deep circuitry in case n=0,1,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x UInt to PopCount
    * @return true.B when x has exactly n bits set
    */
  def equalTo(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): Bool = {
    require(n >= 0, "Cannot check for negative number of bits")
    n match {
      case h: Int if h > x.getWidth => false.B
      case h: Int if h <= 1         => atLeast(n, x) && !greaterThan(n, x)
      case h: Int if h >= x.getWidth - 1 =>
        equalTo(x.getWidth - h, ~x) // check one bit NOT set instead of all-but-one set
      case _ => PopCount(x) === n.U
    }
  }

  /** Implements PopCount(x)==n with less deep circuitry in case n=0,1,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x Seq/Vec of Bool to PopCount
    * @return true.B when x has exactly n bits set
    */
  def equalTo(n: Int, x: Iterable[Bool])(implicit sourceInfo: SourceInfo): Bool = equalTo(n, VecInit(x.toSeq).asUInt)

  /** Implements PopCount(x)>n with less deep circuitry in case n=0,1,x.width-1
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x UInt to PopCount
    * @return true.B when x has more than n bits set
    */
  def greaterThan(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): Bool = {
    require(n >= 0, "Cannot check for negative number of bits")
    atLeast(n + 1, x)
  }

  /** Implements PopCount(x)>n with less deep circuitry in case n=0,1,x.width-1
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x Seq/Vec of Bool to PopCount
    * @return true.B when x has more than n bits set
    */
  def greaterThan(n: Int, x: Iterable[Bool])(implicit sourceInfo: SourceInfo): Bool =
    greaterThan(n, VecInit(x.toSeq).asUInt)

  /** Implements PopCount(x)>=n with less deep circuitry in case n=0,1,2,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x UInt to PopCount
    * @return true.B when x has n or more bits set
    */
  def atLeast(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): Bool = {
    require(n >= 0, "Cannot check for negative number of bits")
    n match {
      case 0 => true.B
      case 1 => x.orR
      case 2 => (x & (x - 1.U)) > 0.U
      case h: Int if h == x.getWidth - 1 => x.andR || equalTo(1, ~x)
      case h: Int if h == x.getWidth     => x.andR
      case h: Int if h > x.getWidth      => false.B
      case _ => PopCount(x) >= n.U
    }
  }

  /** Implements PopCount(x)>=n with less deep circuitry in case n=0,1,2,x.width-1,x.width
    * @param n Int  Static value that PopCount(x) is compared against
    * @param x Seq/Vec of Bool to PopCount
    * @return true.B when x has n or more bits set
    */
  def atLeast(n: Int, x: Iterable[Bool])(implicit sourceInfo: SourceInfo): Bool = atLeast(n, VecInit(x.toSeq).asUInt)
}

/** Create repetitions of the input using a tree fanout topology.
  *
  * @example {{{
  * Fill(2, "b1000".U)  // equivalent to "b1000 1000".U
  * Fill(2, "b1001".U)  // equivalent to "b1001 1001".U
  * Fill(2, myUIntWire)  // dynamic fill
  * }}}
  */
object Fill extends Fill$Intf {

  protected def _applyImpl(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): UInt = {
    n match {
      case _ if n < 0 => throw new IllegalArgumentException(s"n (=$n) must be nonnegative integer.")
      case 0          => UInt(0.W)
      case 1          => x
      case _ if x.isWidthKnown && x.getWidth == 1 =>
        Mux(x.asBool, ((BigInt(1) << n) - 1).asUInt(n.W), 0.U(n.W))
      case _ =>
        val nBits = log2Ceil(n + 1)
        val p2 = Array.ofDim[UInt](nBits)
        p2(0) = x
        for (i <- 1 until p2.length)
          p2(i) = Cat(p2(i - 1), p2(i - 1))
        Cat((0 until nBits).filter(i => (n & (1 << i)) != 0).map(p2(_)))
    }
  }
}

/** Returns the input in bit-reversed order. Useful for little/big-endian conversion.
  *
  * @example {{{
  * Reverse("b1101".U)  // equivalent to "b1011".U
  * Reverse("b1101".U(8.W))  // equivalent to "b10110000".U
  * Reverse(myUIntWire)  // dynamic reverse
  * }}}
  */
object Reverse extends Reverse$Intf {

  private def doit(in: UInt, length: Int)(implicit sourceInfo: SourceInfo): UInt =
    length match {
      case _ if length < 0  => throw new IllegalArgumentException(s"length (=$length) must be nonnegative integer.")
      case _ if length <= 1 => in
      case _ if isPow2(length) && length >= 8 && length <= 64 =>
        // This esoterica improves simulation performance
        var res = in
        var shift = length >> 1
        var mask = ((BigInt(1) << length) - 1).asUInt(length.W)
        while ({
          mask = mask ^ (mask(length - shift - 1, 0) << shift)
          res = ((res >> shift) & mask) | ((res(length - shift - 1, 0) << shift) & ~mask)
          shift = shift >> 1
          shift > 0
        }) {}
        res
      case _ =>
        val half = (1 << log2Ceil(length)) / 2
        Cat(doit(in(half - 1, 0), half), doit(in(length - 1, half), length - half))
    }

  protected def _applyImpl(in: UInt)(implicit sourceInfo: SourceInfo): UInt = doit(in, in.getWidth)
}
